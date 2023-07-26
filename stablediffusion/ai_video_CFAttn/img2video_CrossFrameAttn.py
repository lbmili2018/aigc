""" 
---------------------------------------------------------------------------------------------
controlnetvideo.py

Stable Diffusion Video2Video ControlNet Model

by Victor Condino <un1tz3r0@gmail.com>
May 21 2023

This file contains the code for the video2video controlnet model, which can apply Stable
Diffusion to a video, while maintaining frame-to-frame consistency.	It is based on the
Stable Diffusion img2img model, but adds a motion estimator and motion compensator to
maintain consistency between frames.
---------------------------------------------------------------------------------------------
"""

import click
import numpy as np
import torch
import PIL.Image
import cv2
from PIL import Image
from diffusers import ControlNetModel, UniPCMultistepScheduler, EulerAncestralDiscreteScheduler, AutoencoderKL
from stable_diffusion_controlnet_img2img import StableDiffusionControlNetImg2ImgPipeline
from typing import Tuple, List, FrozenSet, Sequence, MutableSequence, Mapping, Optional, Any, Type, Union
from controlnet_aux import CannyDetector, OpenposeDetector, MLSDdetector, HEDdetector, MidasDetector
import pathlib
import time

# ---------------------------------------------------------------------------------------------
# Motion estimation using the RAFT optical flow model (and some legacy
# farneback code that is not currently used)
# ---------------------------------------------------------------------------------------------

from torchvision.models.optical_flow import raft_large
from torchvision.models.optical_flow import Raft_Large_Weights
import torchvision.transforms.functional
import torch
import os, pandas
import numpy as np

## never got this working yet, could significantly improve results
#
# def smooth_flow_spatiotemporal(flow, sigma:float):
# 	''' smooth dense optical flow using a 3D edge-preserving filter. note implementing this
# 	with opencv is quite possible but would require delaying the frame output due to the
# 	nonlocal-means denoising algorithm's working on the middle frame of a temporal window
# 	consisting of an odd number of frames so as to be symmetrical in lookahead and previously
# 	seen frames. '''
#
# 	cv2.fastNlMeansDenoisingMulti(flow, flow, sigma, 0, 0)

# -----------------------------------------------------------------------------------------------
# helpers for depth-based controlnets
# -----------------------------------------------------------------------------------------------

class MidasDetectorWrapper:
	''' a wrapper around the midas detector model which allows
	choosing either the depth or the normal output on creation '''
	def __init__(self, output_index=0, **kwargs):
		self.model = MidasDetector()
		self.output_index = output_index
		self.default_kwargs = dict(kwargs)
	def __call__(self, image, **kwargs):
		ka = dict(list(self.default_kwargs.items()) + list(kwargs.items()))
		#return torch.tensor(self.model(np.asarray(image), **ka)[self.output_index][None, :, :].repeat(3,0)).unsqueeze(0)
		return PIL.Image.fromarray(self.model(np.asarray(image), **ka)[self.output_index]).convert("RGB")


def depth_to_normal(image):
		''' converts 2d 1ch (z) grayscale depth map image
							to 2d 3ch (xyz) surface normal map image using sobel filter and cross product '''
		image_depth = image.copy()
		image_depth -= np.min(image_depth)
		image_depth /= np.max(image_depth)

		bg_threshold = 0.1

		x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
		x[image_depth < bg_threshold] = 0

		y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
		y[image_depth < bg_threshold] = 0

		z = np.ones_like(x) * np.pi * 2.0

		image = np.stack([x, y, z], axis=2)
		image /= np.sum(image ** 2.0, axis=2, keepdims=True) ** 0.5
		image = (image * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
		image = Image.fromarray(image)
		return image

# -----------------------------------------------------------------------------------------------
# general image helpers (could be moved to a separate file)
# -----------------------------------------------------------------------------------------------

def padto(image, width, height, gravityx=0.5, gravityy=0.5):
	import PIL.ImageOps, PIL.Image
	''' pad image to width and height '''
	image = PIL.Image.fromarray(image)
	if image.size[0] < width:
		image = PIL.ImageOps.expand(image, border=(int((width - image.size[0]) / 2), 0, 0, 0), fill=0)
	if image.size[1] < height:
		image = PIL.ImageOps.expand(image, border=(0, int((height - image.size[1]) / 2), 0, 0), fill=0)
	return image

def topil(image):
	# convert to PIL.Image.Image from various types
	if isinstance(image, PIL.Image.Image):
		return image
	elif isinstance(image, np.ndarray):
		return PIL.Image.fromarray(image)
	elif isinstance(image, torch.Tensor):
		while image.ndim > 3 and image.shape[0] == 1:
			image = image[0]
		if image.ndim == 3 and image.shape[0] in [3, 1]:
			image = image.permute(1, 2, 0)
		return PIL.Image.fromarray(image.numpy())
	else:
		raise ValueError(f"cannot convert {type(image)} to PIL.Image.Image")

def stackh(images):
	''' stack images horizontally, using the largest image's height for all images '''
	images = [topil(image) for image in images]
	cellh = max([image.height for image in images])
	cellw = max([image.width for image in images])
	result = PIL.Image.new("RGB", (cellw * len(images), cellh), "black")
	for i, image in enumerate(images):
		result.paste(image.convert("RGB"), (i * cellw + (cellw - image.width)//2, (cellh - image.height)//2))
	#print(f"stackh: {len(images)}, cell WxH: {cellw}x{cellh}, result WxH: {result.width}x{result.height}")
	return result

def expanddims(*sides):
	''' takes an array of 1, 2, or 4 floating point numbers, which are interpreted
	as a single value, a horizontal and vertical value, or a top, right, bottom, left value
	and returns an array of top, right, bottom and left values, with the same value repeated
	if only one value is given, or the same values repeated twice if two values are given.
	if gravity is given, it must be either one or two values, and is used to offset the '''
	from typing import Tuple, Iterable, ByteString
	if not (isinstance(sides, Iterable) and not isinstance(sides, (str, ByteString))):
		sides = [sides]
	if len(sides) == 1:
		sides = [sides[0], sides[0], sides[0], sides[0]]
	if len(sides) == 2:
		sides = [sides[0], sides[1], sides[0], sides[1]]
	if len(sides) == 3:
		sides = [sides[0], sides[1], sides[2], sides[1]]
	return sides

def roundrect(size, radius:Tuple[int,int,int,int], border:Tuple[int,int,int,int], fill="white", outline="black"):
	from PIL import Image, ImageDraw
	width, height = size
	tl, tr, br, bl = radius
	tl = min(tl, width//2, height//2)
	tr = min(tr, width//2, height//2)
	bl = min(bl, width//2, height//2)
	br = min(br, width//2, height//2)
	btl, btr, bbr, bbl = border
	btl = min(btl, width//2, height//2)
	btr = min(btr, width//2, height//2)
	bbl = min(bbl, width//2, height//2)
	bbr = min(bbr, width//2, height//2)
	result = PIL.Image.new("RGBA", size, color=fill)
	draw = ImageDraw.Draw(result)
	draw.rectangle((0,0,width,height), fill=fill, outline=outline)
	draw.rectangle((btl, btl, width-btr, height-bbr), fill=None, outline=outline)
	draw.rectangle((bbl, bbl, width-bbr, height-bbr), fill=None, outline=outline)
	draw.pieslice((0, 0, tl*2, tl*2), 180, 270, fill=None, outline=outline)
	draw.pieslice((width-tr*2, 0, width, tr*2), 270, 360, fill=None, outline=outline)
	draw.pieslice((0, height-bl*2, bl*2, height), 90, 180, fill=None, outline=outline)
	draw.pieslice((width-br*2, height-br*2, width, height), 0, 90, fill=None, outline=outline)
	return result

def textbox(s, font, color, padding=(1,1,1,1), border=(0,0,0,0), corner_radius=(2,2,2,2), background_color="white", border_color="black"):
	import PIL.Image, PIL.ImageDraw
	def fontgetsize(s):
		draw=PIL.ImageDraw.Draw(PIL.Image.new('RGBA', (1,1), background_color))
		return draw.textsize(s, font=font)
	text = PIL.Image.new('RGBA', fontgetsize(s), background_color)
	draw = PIL.ImageDraw.Draw(text)
	draw.text((0, 0), s, font=font, fill=color)
	return text

# ---------------------------------------------------------------------------------------------
# Colorspace conversion extravaganza (these belong in a colorspace.py file for sure)
# ---------------------------------------------------------------------------------------------

def rgbtohsl(rgb:np.ndarray):
	''' vectorized rgb to hsl conversion
	input is a numpy array of shape (..., 3) and dtype float32 or uint8'''
	if rgb.dtype == np.uint8:
		rgb = rgb.astype(np.float32) / 255.0
	r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
	#r, g, b = r / 255.0, g / 255.0, b / 255.0
	mx = np.amax(rgb, 2)
	mn = np.amin(rgb, 2)
	df = mx-mn
	h = np.zeros(r.shape)
	h[g > r] = (60 * ((g[g>r]-b[g>r])/df[g>r]) + 360) % 360
	h[b > g] = (60 * ((b[b>g]-r[b>g])/df[b>g]) + 240) % 360
	h[r > b] = (60 * ((r[r>b]-g[r>b])/df[r>b]) + 120) % 360
	# h[r == g == b] = 0
	s = np.zeros(r.shape)
	s[np.nonzero(mx)] = df[np.nonzero(mx)]/mx[np.nonzero(mx)]
	l = np.zeros(r.shape)
	l = (mx+mn)/2
	hsl = np.zeros(rgb.shape)
	hsl[:,:,0] = h
	hsl[:,:,1] = s
	hsl[:,:,2] = l
	return hsl #np.ndarray([h, s, l])

def hsltorgb(hsl:np.ndarray):
	''' vectorized hsl to rgb conversion
	input is a numpy array of shape (..., 3) and dtype float, with hue first in 0-360, then sat and lum in 0-1 '''
	h, s, l = hsl[:,:,0], hsl[:,:,1], hsl[:,:,2]
	c = (1 - np.abs(2*l-1)) * s
	h = h / 60
	x = c * (1 - np.abs(h % 2 - 1))
	m = l - c/2
	r = np.zeros(h.shape)
	g = np.zeros(h.shape)
	b = np.zeros(h.shape)
	r[h < 1] = c[h < 1]
	r[h >= 1] = x[h >= 1]
	g[h < 1] = x[h < 1]
	g[h >= 2] = c[h >= 2]
	b[h < 2] = c[h < 2]
	b[h >= 3] = x[h >= 3]
	r[h >= 4] = c[h >= 4]
	g[h >= 4] = x[h >= 4]
	r += m
	g += m
	b += m
	r *= 255
	g *= 255
	b *= 255
	return np.ndarray([r, g, b])

def hsltorgb(hsl:np.ndarray):
  h, s, l = hsl[:,:,0], hsl[:,:,1], hsl[:,:,2]
  c = (1 - np.abs(2*l-1)) * s
  h = h / 60
  x = c * (1 - np.abs(h % 2 - 1))
  m = l - c/2
  r = np.zeros(h.shape)
  g = np.zeros(h.shape)
  b = np.zeros(h.shape)
  r[h < 1] = c[h < 1]
  r[h >= 1] = x[h >= 1]
  g[h < 1] = x[h < 1]
  g[h >= 2] = c[h >= 2]
  b[h < 2] = c[h < 2]
  b[h >= 3] = x[h >= 3]
  r[h >= 4] = c[h >= 4]
  g[h >= 4] = x[h >= 4]
  r += m
  g += m
  b += m
  r *= 255
  g *= 255
  b *= 255
  return np.ndarray([r, g, b])

def brightcontrastmatch(source, template):
	oldshape = source.shape
	source = source.ravel()
	template = template.ravel()
	s_mean, s_std = source.mean(), source.std()
	t_mean, t_std = template.mean(), template.std()
	source = (source - s_mean) * (t_std / s_std) + t_mean
	return source.reshape(oldshape)

def avghuesatmatch(source, template):
	source = np.asarray(source)
	template = np.asarray(template)
	oldshape = source.shape
	source = source.ravel()
	template = template.ravel()
	s_hsl = np.asarray(PIL.Image.fromarray(source, mode="RGB").convert(mode="HSL"))
	t_hsl = np.asarray(PIL.Image.fromarray(template, mode="RGB").convert(mode="HSL"))
	s_hue, s_sat = s_hsl[:,0], s_hsl[:,1]
	t_hue, t_sat = t_hsl[:,0], t_hsl[:,1]
	s_hue_mean, s_hue_std = s_hue.mean(), s_hue.std()
	s_sat_mean, s_sat_std = s_sat.mean(), s_sat.std()
	t_hue_mean, t_hue_std = t_hue.mean(), t_hue.std()
	t_sat_mean, t_sat_std = t_sat.mean(), t_sat.std()
	s_hue = (s_hue - s_hue_mean) * (t_hue_std / s_hue_std) + t_hue_mean
	s_sat = (s_sat - s_sat_mean) * (t_sat_std / s_sat_std) + t_sat_mean
	s_hsl[:,0], s_hsl[:,1] = s_hue, s_sat
	return (PIL.Image.fromarray(s_hsl.reshape(oldshape), mode="HSL")).convert(mode="RGB")

def histomatch(source, template):
	oldshape = source.shape
	source = source.ravel()
	template = template.ravel()
	s_values, bin_idx, s_counts = np.unique(source, return_inverse=True, return_counts=True)
	t_values, t_counts = np.unique(template, return_counts=True)
	s_quantiles = np.cumsum(s_counts).astype(np.float64)
	s_quantiles /= s_quantiles[-1]
	t_quantiles = np.cumsum(t_counts).astype(np.float64)
	t_quantiles /= t_quantiles[-1]
	interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
	return interp_t_values[bin_idx].reshape(oldshape)

from skimage.exposure import match_histograms
import cv2

def maintain_colors(color_match_sample, prev_img, mode, amount=1.0):
		''' adjust output frame to match histogram of first output frame,
		this is how deforum does it, big thanks to them '''
		
		if mode == 'rgb':
				return match_histograms(prev_img, color_match_sample, multichannel=True)
		elif mode == 'hsv':
				prev_img_hsv = cv2.cvtColor(prev_img, cv2.COLOR_RGB2HSV)
				color_match_hsv = cv2.cvtColor(color_match_sample, cv2.COLOR_RGB2HSV)
				matched_hsv = match_histograms(prev_img_hsv, color_match_hsv, multichannel=True)
				return cv2.cvtColor(matched_hsv, cv2.COLOR_HSV2RGB)
		elif mode == 'lab':
				prev_img_lab = cv2.cvtColor(prev_img, cv2.COLOR_RGB2LAB)
				color_match_lab = cv2.cvtColor(color_match_sample, cv2.COLOR_RGB2LAB)
				matched_lab = match_histograms(prev_img_lab, color_match_lab, multichannel=True)
				return cv2.cvtColor(matched_lab, cv2.COLOR_LAB2RGB)
		else:
				raise ValueError('Invalid color mode')

# ----------------------------------------------------------------------------------------------

def process_frames(input_video, output_video, wrapped, start_time=None, end_time=None, duration=None, max_dimension=None, min_dimension=None, round_dims_to=None, fix_orientation=False, ):
		from moviepy import editor as mp
		from PIL import Image
		from tqdm import tqdm
		# Load the video
		video = mp.VideoFileClip(input_video)
		orig_w, orig_h = video.size
		print(10, video.size)
		# scale the video frames if a min/max size is given
		if fix_orientation:
			w, h = video.size
		else:
			w, h = video.size
		print(11, w, h)
		if max_dimension != None:
			if w > h:
				w, h = max_dimension, int(h / w * max_dimension)
			else:
				w, h = int(w / h * max_dimension), max_dimension

		print(12, w, h)
		if min_dimension != None:
			if w < h:
				w, h = min_dimension, int(h / w * min_dimension)
			else:
				w, h = int(w / h * min_dimension), min_dimension
		print(13, w, h)
		if round_dims_to is not None:
			w = round_dims_to * (w // round_dims_to)
			h = round_dims_to * (h // round_dims_to)

		print(14, w, h)
		# set the start and end time and duration to process if given
		if end_time is not None:
			video = video.subclip(0, end_time)
		if start_time is not None:
			video = video.subclip(start_time)
		if duration != None:
			video = video.subclip(0, duration)

		# Create a new video with the processed frames
		from time import monotonic
		try:
			framenum = 0
			starttime = monotonic()
			def wrapper(gf, t):
				start = time.time()
				nonlocal framenum
				nonlocal starttime
				elapsed = monotonic() - starttime
				if t > 0:
					eta = (video.duration / t) * elapsed
				else:
					eta = 0
				print(f"Processing frame {framenum} at time {t}/{video.duration} seconds... {elapsed:.2f}s elapsed, {eta:.2f}s estimated time remaining")
				result = wrapped(framenum, PIL.Image.fromarray(gf(t)).resize((w,h)))
				# result = result.resize((orig_w, orig_h))
				print("everyone", result.width, result.height)
				framenum = framenum + 1

				print("total running time: {}".format(time.time() - start))
				return np.asarray(result)

			with open(output_video.replace("mp4", "txt"), "w") as f:
				f.write("{}_{}".format(orig_w, orig_h))  # 自带文件关闭功能，不需要再写
			f.close()
			#video.fx(wrapper).write_videofile(output_video)
			video.fl(wrapper, keep_duration=True).write_videofile(output_video)

		#processed_video = mp.ImageSequenceClip([
			#  np.array(wrapped(framenum, Image.fromarray(frame).resize((w,h))))
			#    for framenum, frame in
			#      enumerate(tqdm(video.iter_frames()))
			#  ], fps=video.fps)
		finally:
			# save the video
			#if processed_video != None:
			#  processed_video.write_videofile(output_video)
			video.close()


def frame2video(im_dir, video_dir, fps):
	im_list = os.listdir(im_dir)
	# im_list.sort(key =lambda x: int(x.split('_')[1]))  # key=lambda x: int(x.replace("frame", "").split('.')[0])
	im_list.sort()
	print("im_list:", im_list)

	img = Image.open(os.path.join(im_dir, im_list[0]))
	img_size = img.size
	fourcc = cv2.VideoWriter_fourcc(*'XVID')  # opencv版本是3
	videoWriter = cv2.VideoWriter(video_dir, fourcc, fps, img_size)
	# count = 1
	for i in im_list:
		im_name = os.path.join(im_dir + i)
		frame = cv2.imdecode(np.fromfile(im_name, dtype=np.uint8), -1)
		videoWriter.write(frame)
	videoWriter.release()
	print('merge new video successfully')

def video2frame(videos_path, frames_save_path, time_interval):
    if not os.path.exists(frames_save_path):
        os.mkdir(frames_save_path)
    vidcap = cv2.VideoCapture(videos_path)
    success, image = vidcap.read()
    count = 0
    while success:
        try:
            success, image = vidcap.read()
            # print("image:", image.shape, ("%d" % count).zfill(6))
            count += 1
            if count % time_interval == 0:
                cv2.imwrite(frames_save_path + '/{}.png'.format(str(count).zfill(6)), image)  # 存储为图像
        except:
            pass
    print("extract frames interval:", time_interval)
    print("after extract nums:", count // time_interval)

# ----------------------------------------------------------------------------------------------
# Main entry point, look at all those options!
# ----------------------------------------------------------------------------------------------

@click.command()
# input and output video arguments
@click.argument('input_video', type=click.Path(exists=True))
@click.argument('output_video', type=click.Path())
# video timeline options
@click.option('--start-time', type=float, default=None, help="start time in seconds")
@click.option('--end-time', type=float, default=None, help="end time in seconds")
@click.option('--duration', type=float, default=None, help="duration in seconds")
# video scaling options
@click.option('--max-dimension', type=int, default=832, help="maximum dimension of the video")
@click.option('--min-dimension', type=int, default=512, help="minimum dimension of the video")
@click.option('--round-dims-to', type=int, default=128, help="round the dimensions to the nearest multiple of this number")
@click.option('--fix-orientation/--no-fix-orientation', is_flag=True, default=True, help="resize videos shot in portrait mode on some devices to fix incorrect aspect ratio bug")
# not implemented... yet (coming soon: beat-reactive video processing):
@click.option('--no-audio', is_flag=True, default=False, help="don't include audio in the output video, even if the input video has audio")
@click.option('--audio-from', type=click.Path(exists=True), default=None, help="audio file to use for the output video, replaces the audio from the input video, will be truncated to duration of input or --duration if given")
@click.option('--audio-offset', type=float, default=None, help="offset in seconds to start the audio from, when used with --audio-from")
# stable diffusion options
@click.option('--prompt', type=str, default=None, help="prompt used to guide the denoising process")
@click.option('--negative-prompt', type=str, default=None, help="negative prompt, can be used to prevent the model from generating certain words")
@click.option('--prompt-strength', type=float, default=7.5, help="how much influence the prompt has on the output")
#@click.option('--scheduler', type=click.Choice(['default']), default='default', help="which scheduler to use")
@click.option('--num-inference-steps', '--steps', type=int, default=25, help="number of inference steps, depends on the scheduler, trades off speed for quality. 20-50 is a good range from fastest to best.")
@click.option('--controlnet', type=click.Choice(['aesthetic', 'lineart21', 'hed', 'hed21', 'canny', 'canny21', 'openpose', 'openpose21', 'depth', 'depth21', 'normal', 'mlsd']), default='hed', help="which pretrained controlnet annotator to use")
@click.option('--controlnet-strength', type=float, default=1.0, help="how much influence the controlnet annotator's output is used to guide the denoising process")
@click.option('--init-image-strength', type=float, default=0.5, help="the init-image strength, or how much of the prompt-guided denoising process to skip in favor of starting with an existing image")
@click.option('--feedthrough-strength', type=float, default=0.0, help="the ratio of input to motion compensated prior output to feed through to the next frame")
@click.option('--sd-model', type=str, default=None)
@click.option('--sd-vae', type=str, default=None)

# motion smoothing options
@click.option('--motion-alpha', type=float, default=0.1, help="smooth the motion vectors over time, 0.0 is no smoothing, 1.0 is maximum smoothing")
@click.option('--motion-sigma', type=float, default=0.3, help="smooth the motion estimate spatially, 0.0 is no smoothing, used as sigma for gaussian blur")
# debugging/progress options
@click.option('--show-detector/--no-show-detector', is_flag=True, default=False, help="show the controlnet detector output")
@click.option('--show-input/--no-show-input', is_flag=True, default=False, help="show the input frame")
@click.option('--show-output/--no-show-output', is_flag=True, default=True, help="show the output frame")
@click.option('--show-motion/--no-show-motion', is_flag=True, default=False, help="show the motion transfer (not implemented yet)")
@click.option('--dump-frames', type=click.Path(), default=None, help="write intermediate frame images to a file/files during processing to visualise progress. may contain various {} placeholders")
@click.option('--skip-dumped-frames', is_flag=True, default=False, help="read dumped frames from a previous run instead of processing the input video")
@click.option('--dump-video', is_flag=True, default=False, help="write intermediate dump images to the final video instead of just the final output image")
# Color-drift fixing options
@click.option('--color-fix', type=click.Choice(['none', 'rgb', 'hsv', 'lab']), default='lab', help="prevent color from drifting due to feedback and model bias by fixing the histogram to the first frame. specify colorspace for histogram matching, e.g. 'rgb' or 'hsv' or 'lab', or 'none' to disable.")
@click.option('--color-amount', type=float, default=0.0, help="blend between the original color and the color matched version, 0.0-1.0")
@click.option('--color-info', is_flag=True, default=False, help="print extra stats about the color content of the output to help debug color drift issues")
# Detector-specific options
@click.option('--canny-low-thr', type=float, default=100, help="canny edge detector lower threshold")
@click.option('--canny-high-thr', type=float, default=200, help="canny edge detector higher threshold")
@click.option('--mlsd-score-thr', type=float, default=0.1, help="mlsd line detector v threshold")
@click.option('--mlsd-dist-thr', type=float, default=0.1, help="mlsd line detector d threshold")
def main(input_video, output_video, start_time, end_time,
		sd_model, sd_vae,
	 	duration, max_dimension, min_dimension, round_dims_to, 
		no_audio, audio_from, audio_offset, prompt, negative_prompt, 
		prompt_strength, num_inference_steps, controlnet, 
		controlnet_strength, fix_orientation, init_image_strength, 
		feedthrough_strength, motion_alpha, motion_sigma, 
		show_detector, show_input, show_output, show_motion, 
		dump_frames, skip_dumped_frames, dump_video, 
		color_fix, color_amount, color_info, canny_low_thr=None, 
		canny_high_thr=None, mlsd_score_thr=None, mlsd_dist_thr=None,
	):
	
	# substitute {} placeholders in output_video with input_video basename
	if output_video != None and input_video != None:
			inputpath=pathlib.Path(input_video).resolve()
			output_video = output_video.format(inpath=str(inputpath), indir=str(inputpath.parent), instem=str(inputpath.stem))
			output_video_path = pathlib.Path(output_video)
			if not output_video_path.parent.exists():
				output_video_path.parent.mkdir(parents=True)
			output_video = str(output_video_path)
	
	# run controlnet pipeline on video
	# choose controlnet model and detector based on the --controlnet option
	# this also affects the default scheduler and the stable diffusion model version required, in the case of aesthetic controlnet
	scheduler = 'eulera'
	sdmodel = sd_model
	sdvae = sd_vae
	vae = AutoencoderKL.from_pretrained(sdvae, subfolder="vae", torch_dtype=torch.float16).to("cuda")

	if controlnet == 'canny':
		detector_kwargs = dict({
			"low_threshold": canny_low_thr if canny_low_thr != None else 50,
			"high_threshold": canny_high_thr if canny_high_thr != None else 200
		})
		detector_model = CannyDetector()
		controlnet_model = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
	elif controlnet == 'depth':
		detector_kwargs = dict()
		detector_model = MidasDetectorWrapper()
		controlnet_model = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16)
	elif controlnet == 'hed':
		detector_kwargs = dict()
		detector_model = HEDdetector.from_pretrained("lllyasviel/ControlNet")
		controlnet_model = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-hed", torch_dtype=torch.float16)
	elif controlnet == 'hed21':
		detector_kwargs = dict()
		detector_model = HEDdetector.from_pretrained("lllyasviel/ControlNet")
		controlnet_model = ControlNetModel.from_pretrained("thibaud/controlnet-sd21-hed-diffusers", torch_dtype=torch.float16)
		#sdmodel = 'stabilityai/stable-diffusion-2-1'
	else:
		raise NotImplementedError("controlnet type not implemented")

	# instantiate the diffusion pipeline
	if controlnet_model != None:
		pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(sdmodel, controlnet=controlnet_model, torch_dtype=torch.float16)
	else:
		pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(sdmodel, torch_dtype=torch.float16)

	# set the scheduler... this is a bit hacky but it works
	if scheduler == 'unipcm':
		pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
	elif scheduler == 'eulera':
		pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

	pipe.vae = vae
	pipe.enable_xformers_memory_efficient_attention()
	pipe.enable_model_cpu_offload()
	pipe.run_safety_checker = lambda image, text_embeds, text_embeds_negative: (image, False)

	# nonlocal first_output_frame
	W, H = 512, 512
	# videos_path = "/root/limiao/controlnetvideo_test/user/black.mp4"
	frames_save_path = "/root/limiao/controlnetvideo_CFAttn/user/temp/frame/"
	# video2frame(videos_path=videos_path,frames_save_path=frames_save_path, time_interval=1)
	imgs = os.listdir(frames_save_path)
	imgs.sort()
	print("imgs:", imgs[:100])
	img_list = []
	for x in imgs:
		img_list.append(Image.open(frames_save_path + x).convert("RGB").resize((W, H)))

	pre_frame = None
	anchor_frame = None
	strength = feedthrough_strength * init_image_strength
	print("strength:", strength)
	for i in range(len(img_list)):
		generator = torch.Generator(device='cuda').manual_seed(1024)
		if i==0:
			anchor_frame = img_list[0]
			former_frame = img_list[0]
			curr_frame = img_list[0]
		elif i==1:
			former_frame = img_list[0]
			curr_frame = img_list[i]
		else:
			anchor_frame = img_list[0]
			former_frame = img_list[i-1]
			curr_frame = img_list[i]
		input_image = [anchor_frame, former_frame, curr_frame]

		# input_image = [img_list[i]] * 3

		width, height = input_image[0].size
		print("2 {}, {}".format(width, height))
		input_frame = np.asarray(input_image)
		print("input_frame:", input_frame.size)

		control_image = []
		for xx in input_image:
			ss = detector_model(xx.resize((W, H)), **detector_kwargs).convert("RGB")
			print("ss:", ss.size)
			control_image.append(ss)

		init_image = input_image.copy()
		# run the pipeline
		a = time.time()
		output_frame = pipe(
			prompt=[prompt] * 3,
			negative_prompt=[negative_prompt] * 3,
			guidance_scale=prompt_strength,
			num_inference_steps=num_inference_steps,
			generator=generator,
			controlnet_conditioning_image=control_image,
			controlnet_conditioning_scale=controlnet_strength,
			image = init_image,
			strength = 1.0 - strength,
			num_images_per_prompt=1,
		).images
		print("infer time: {}".format(time.time() - a))
		print("output_frame:", len(output_frame))


		# print("color fix", color_fix, color_amount)
		# if color_fix != 'none' and color_amount > 0.0:
		# 	if 1: #first_output_frame == None:
		# 		# save the first output frame for color correction
		# 		first_output_frame = output_frame.copy()
		# 	else:
		# 		# skipped frames don't get color correction, since they already have it applied
		# 		if not skipped:
		# 			image = PIL.Image.fromarray(maintain_colors(np.asarray(first_output_frame), np.asarray(output_frame), color_fix)*color_amount).astype(np.uint8)
		# 			image.save("frame.png")
		# 			print("save image successful")
		# 			# blend the color fix into the output frame
		# 			output_frame = PIL.Image.fromarray((
		# 				np.asarray(output_frame)*(1.0-color_amount) +
		# 				maintain_colors(np.asarray(first_output_frame), np.asarray(output_frame), color_fix)*color_amount
		# 			).astype(np.uint8))


		output_frame[-1].save(output_video + "/img_{}.png".format(str(i).zfill(6)))

if __name__ == "__main__":
	main()

	# im_dir = "/root/limiao/controlnetvideo_test/user/temp/black_res11_075_all/"
	# video_dir = "/root/limiao/controlnetvideo_test/user/temp/black_res11_075_crossAttn.mp4"
	# frame2video(im_dir, video_dir, fps=24)