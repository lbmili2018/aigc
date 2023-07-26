#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import cv2
import numpy as np

def composite_image(first, second, var):
	first_bin = first
	first = first /255.0
	second = second / 255.0

	gray_first = cv2.cvtColor(first_bin, cv2.COLOR_BGR2GRAY)
	ret, first_bin = cv2.threshold(gray_first, var, 255, cv2.THRESH_BINARY)
	first_bin = cv2.cvtColor(first_bin, cv2.COLOR_GRAY2RGB)

	first_bin = cv2.bitwise_not(first_bin)
	# cv2.imshow('first_bin', first_bin)
	# cv2.waitKey(0)

	first_bin = first_bin / 255.0
	new = first * (first_bin) + (1-first_bin) * second
	new = (new * 255).astype('uint8')

	# cv2.imshow('new', new)
	# cv2.waitKey(0)

	return new

def get_zc_mask(frame):
	gray_logo = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	ret, mask_bin = cv2.threshold(gray_logo, 20, 255, cv2.THRESH_BINARY)
	mask = cv2.cvtColor(mask_bin, cv2.COLOR_GRAY2RGB)
	return mask / 255.0


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


def main(orig_vide, res_video, out_dir):
	if fast_vaild:
		origs = load_gif_file(path=orig_vide, union_size=(256, 256))
		ress = load_gif_file(path=res_video, union_size=(256, 256))
	else:
		origs = load_gif_file(path=orig_vide)
		ress = load_gif_file(path=res_video, union_size=(origs[0].shape[1], origs[0].shape[0]))
	print("all origs:", len(origs))
	print("all ress:", len(ress))


	zc_start = 150
	zc_during = 30  #fixed
	assert zc_start > zc_during
	assert (zc_start + zc_during) < (len(origs) - zc_during)

	new_frames = []
	idx = 0
	for i in range(zc_start, zc_start + zc_during):
		first = origs[i]
		second = ress[i]
		new_frame = composite_image(first, second, 255 - idx* int(255/zc_during))
		# cv2.imshow('new_frame', new_frame)
		# cv2.waitKey(0)
		idx +=1
		new_frames.append(new_frame)
	print("len(new_frames):", len(new_frames))

	new_video = origs[:zc_start]
	new_video.extend(new_frames)
	new_video.extend(ress[zc_start + zc_during:])
	final = np.array(new_video)
	print("final:", final.shape)

	W, H = origs[0].shape[1], origs[0].shape[0]
	fps = 24
	size = (W, H * 1)  # (1440, 960)
	video = cv2.VideoWriter(out_dir,cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)
	for i in range(len(final)):
		video.write(final[i])

	video.release()
	cv2.destroyAllWindows()


if __name__ == '__main__':
	fast_vaild = True
	orig_vide = '/root/limiao/input.mp4'
	res_video = '/root/limiao/output.mp4'
	out_dir = "/dnwx/datasets/sam/share/zhijie.avi"
	main(orig_vide, res_video, out_dir)

