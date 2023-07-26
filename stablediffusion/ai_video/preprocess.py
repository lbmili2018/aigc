import numpy as np
import pandas as pd
from PIL import Image
import os
import cv2
import argparse

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

def frame2video(im_dir, video_dir, fps):
    im_list = os.listdir(im_dir)
    im_list.sort()  # key=lambda x: int(x.replace("frame", "").split('.')[0])
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


def merge_video(data_path1, data_path2, out_dir, fps, data_path3=None):
    im_list1 =[x for x in os.listdir(data_path1) if x.endswith('.jpg') or x.endswith('.png')]
    im_list1.sort()
    print("im_list111:", len(im_list1))

    im_list2 =[x for x in os.listdir(data_path2) if x.endswith('.jpg') or x.endswith('.png')]
    im_list2.sort()
    print("im_list222:", len(im_list2))
    W, H, _ = cv2.imread(data_path2 + im_list2[0]).shape
    size = (H*2, W) # (1440, 960)
    if data_path3:
        im_list3 =[x for x in os.listdir(data_path3) if x.endswith('.jpg') or x.endswith('.png')]
        im_list3.sort()
        print("im_list333:", len(im_list3))
        size = (H*3, W)
    video = cv2.VideoWriter(out_dir, cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), fps, size)

    for i in range(len(im_list2)):
        image = data_path1 + im_list1[i]
        res2 = data_path2 + im_list2[i]

        image = cv2.imread(image)
        cv2.putText(image, "orig", (5,50 ), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        res2 = cv2.imread(res2)
        cv2.putText(res2, "RIFE_2X", (5,50 ), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        if data_path3:
            res3 = data_path3 + im_list3[i]
            res3 = cv2.imread(res3)
            cv2.putText(res3, "RIFE_2X", (5,50 ), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            img_total = np.concatenate((image, res2, res3), axis = 1)
        else:
            img_total = np.concatenate((image, res2), axis = 1)
        # print("img_total:", img_total.shape)
        # cv2.imshow('input img', img_total)
        # cv2.waitKey()
        video.write(img_total)

    video.release()
    cv2.destroyAllWindows()



def setup_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument("--videos_path", type=str, default=None )
	parser.add_argument("--frames_save_path", type=str, default=None )
	parser.add_argument("--time_interval", type=int, default=1 )
	parser.add_argument("--fps", type=int, default=24 )

	parser.add_argument("--video_save_path", type=str, default=None )

	parser.add_argument("--special_effect", type=str, default="bowen_zc", choices=["bowen_zc", "xuhua_zc"])
	parser.add_argument("--zc_time1", type=float, default=0.2 )
	parser.add_argument("--zc_time2", type=float, default=0.6 )
	parser.add_argument("--vis", action="store_true", help="visualization")
	parser.add_argument("--fast_vaild", action="store_true", help="resize 256*256")

	parser.add_argument("--EXTRACT_RIFE", action="store_true", help="visualization")

	return parser


if __name__ == '__main__':
    parser = setup_parser()
    args = parser.parse_args()

    if args.EXTRACT_RIFE:
        video2frame(args.videos_path, args.frames_save_path, args.time_interval)
        frame2video(im_dir=args.frames_save_path + "/", video_dir=args.video_save_path, fps=args.fps)

    else:
        # compare video
        data_path1 = "/root/limiao/controlnetvideo/split/black_out_test_style_vae_035/"
        data_path2 = "/root/limiao/controlnetvideo/split/black_half_out_test_style_vae_035_2X_48fps/"
        data_path3 = "/root/limiao/controlnetvideo/split/black_half_out_test_style_vae_035_2X_48fps/"
        out_dir="/root/limiao/result/OrigBlack_vs_RIFE2X.mp4"
        fps = 24
        merge_video(data_path1, data_path2, out_dir, fps, data_path3)
