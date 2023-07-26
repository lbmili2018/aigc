#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import cv2
import numpy as np

def composite_image(first, second, zc_mask):
	first = first / 255.0
	second = second / 255.0
	mask = zc_mask
	new = first * (1-mask) + mask * second
	# cv2.imshow('new', new)
	# cv2.waitKey(0)
	return (new * 255).astype('uint8')

def get_zc_mask(frame, thres_range=None):
	gray_logo = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	if thres_range:
		ret, mask_bin = cv2.threshold(gray_logo, 20, 255, cv2.THRESH_BINARY)
	else:
		new_mask = gray_logo
		new_mask[gray_logo >= 128] = 0
		new_mask[gray_logo <= 5] = 0
		mask_bin = new_mask
		kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
		mask_bin = cv2.erode(mask_bin, kernel)
		# cv2.imshow("erode", mask_bin)
		kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
		mask_bin = cv2.dilate(mask_bin, kernel)
		# cv2.imshow("dilate", mask_bin)

	mask = cv2.cvtColor(mask_bin, cv2.COLOR_GRAY2RGB) / 255.0
	return mask
	
def load_gif_file(path="input/input_2.MOV", union_size=None):
	cap = cv2.VideoCapture(path)
	res = []
	while (cap.isOpened()):
		ret, frame = cap.read()
		if ret:
			# print("frame:", frame.shape)
			if union_size:
				print("size:", union_size)
				frame = cv2.resize(frame, union_size)
			res.append(frame)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				cap.release()
				break
		else:
			cap.release()

	cv2.destroyAllWindows()
	return res


def main(orig_vide, res_video, zc_video, out_dir):
	if fast_vaild:
		origs = load_gif_file(path=orig_vide, union_size=(256, 256))
		ress = load_gif_file(path=res_video, union_size=(256, 256))
		zcs = load_gif_file(path=zc_video, union_size=(256, 256))
	else:
		origs = load_gif_file(path=orig_vide)
		ress = load_gif_file(path=res_video, union_size=(origs[0].shape[1], origs[0].shape[0]))
		zcs = load_gif_file(path=zc_video, union_size=(origs[0].shape[1], origs[0].shape[0]))
	print("all origs:", len(origs))
	print("all ress:", len(ress))
	print("all zcs:", len(zcs))

	zc_start = 150
	zc_during = len(zcs)  #fixed
	assert zc_start > zc_during
	assert (zc_start + zc_during) < (len(origs) - zc_during)

	new_frames = []
	idx = 0
	for i in range(zc_start, zc_start+zc_during):
		first = origs[i]
		second = ress[i]
		zc_frame =  zcs[idx]

		zc_mask = get_zc_mask(zc_frame,  thres_range=True)
		zc_mask = cv2.resize(zc_mask, (origs[0].shape[1], origs[0].shape[0]))

		new_frame = composite_image(first, second, zc_mask)
		edge_mask = get_zc_mask(zc_frame)
		edge_mask = (edge_mask*255).astype('uint8')
		imgadd = cv2.add(new_frame, edge_mask)

		idx +=1
		new_frames.append(imgadd)

		if vis:
			cv2.imshow('new_frame', new_frame)
			cv2.waitKey(0)

			cv2.imshow('zc_mask', zc_mask)
			cv2.waitKey(0)

			cv2.imshow('edge_mask', edge_mask)
			cv2.waitKey(0)

			cv2.imshow('imgadd', imgadd)
			cv2.waitKey(0)

	print("len(new_frames):", len(new_frames))

	new_video = origs[:zc_start]
	new_video.extend(new_frames)
	new_video.extend(ress[zc_start + zc_during:])


	W, H = origs[0].shape[1], origs[0].shape[0]
	fps = 24
	size = (W, H * 1)  # (1440, 960)
	video = cv2.VideoWriter(out_dir,cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), fps, size)

	final = np.array(new_video)
	print("final:", final.shape)

	for i in range(len(final)):
		video.write(final[i])

	video.release()
	cv2.destroyAllWindows()


if __name__ == '__main__':
	vis = False
	fast_vaild = False
	orig_vide = '/root/limiao/input.mp4'
	res_video = '/root/limiao/output.mp4'
	zc_video = '/root/limiao/333.mp4'
	out_dir = "/dnwx/datasets/sam/share/bowen_720p.mp4"
	main(orig_vide, res_video, zc_video, out_dir)

