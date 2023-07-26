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

def get_zc_mask(frame):
	gray_logo = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	ret, mask_bin = cv2.threshold(gray_logo, 20, 255, cv2.THRESH_BINARY)
	mask = cv2.cvtColor(mask_bin, cv2.COLOR_GRAY2RGB)
	return mask / 255.0
	
def load_gif_file(path="input/input_2.MOV"):
	cap = cv2.VideoCapture(path)
	res = []
	while (cap.isOpened()):
		ret, frame = cap.read()
		if ret:
			# cv2.imshow('frame', frame)
			# print("frame:", frame.shape)
			# frame = cv2.resize(frame, (256, 256))
			res.append(frame)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				cap.release()
				break
		else:
			cap.release()

	cv2.destroyAllWindows()
	return res


def main(orig_vide, res_video, zc_video):
	origs = load_gif_file(path=orig_vide)
	ress = load_gif_file(path=res_video)
	zcs = load_gif_file(path=zc_video)
	print("all origs:", len(origs))
	print("all ress:", len(ress))
	print("all zcs:", len(zcs))

	new_frames = []
	for i, zc in enumerate(zcs[:-5]):
		first = origs[-i]
		second = ress[i]
		zc_frame =  zcs[i]

		zc_mask = get_zc_mask(zc_frame)
		zc_mask = cv2.resize(zc_mask, (origs[0].shape[1], origs[0].shape[0]))
		# print("zc_mask:", zc_mask.shape)
		# cv2.imshow('mask', zc_mask)
		new_frame = composite_image(first, second, zc_mask)
		new_frames.append(new_frame)
	print("len(new_frames):", len(new_frames))

	W, H = 256, 256
	fps = 30
	size = (W, H * 1)  # (1440, 960)
	video = cv2.VideoWriter("/dnwx/datasets/sam/share/test.avi",
							cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)

	new_video = origs[:-len(new_frames)]
	new_video.extend(new_frames)
	new_video.extend(ress[len(new_frames):])
	print("new_video_lens:", len(new_video))

	final = np.array(new_video)
	print("final:", final.shape)

	for i in range(len(final)):
		video.write(final[i])

	video.release()
	cv2.destroyAllWindows()


if __name__ == '__main__':
	orig_vide = '/dnwx/datasets/sam/share/04_hed_guofeng.mp4'
	res_video = '/dnwx/datasets/sam/share/res_video_manhua.mp4'
	# zc_video = '/root/limiao/diffuser/image2video/input/input_1.gif'
	zc_video = '/root/limiao/333.mp4'

	main(orig_vide, res_video, zc_video)

