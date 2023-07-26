import numpy as np
import pandas as pd
from PIL import Image
import os
import cv2

def video2frame(videos_path, frames_save_path, time_interval):
    vidcap = cv2.VideoCapture(videos_path)
    success, image = vidcap.read()
    count = 0
    while success:
        try:
            success, image = vidcap.read()
            print("image:", image.shape, ("%d" % count).zfill(6))
            count += 1
            if count % time_interval == 0:
                cv2.imwrite(r'/root/limiao/test11/{}.jpg'.format(str(count).zfill(6)), image)  # 存储为图像
        except:
            pass
    print(count)


def frame2video(im_dir, video_dir, fps):
    im_list = os.listdir(im_dir)
    im_list.sort()  # key=lambda x: int(x.replace("frame", "").split('.')[0])
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
    print('finish')


def merge_vide(data_path1, data_path2):
    im_list1 =[x for x in os.listdir(data_path1) if x.endswith('.jpg') or x.endswith('.png')]
    im_list1.sort()
    print("im_list111:", len(im_list1))

    im_list2 =[x for x in os.listdir(data_path2) if x.endswith('.jpg') or x.endswith('.png')]
    im_list2.sort()
    print("im_list222:", len(im_list2))
    # W, H, _ = cv2.imread(data_path2 + im_list2[0]).shape
    W, H  = 256, 256
    fps = 20
    size = (256*2, 256) # (1440, 960)
    video = cv2.VideoWriter("/root/limiao/final.avi", cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)
    for i in range(len(im_list1)):
        image = data_path1 + im_list1[i]
        res = data_path2 + im_list2[i]

        image = cv2.imread(image)
        image = cv2.resize(image, (H, W))
        cv2.putText(image, "orig", (5,50 ), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        res = cv2.imread(res)
        res = cv2.resize(res, (H, W))
        cv2.putText(res, "deforum", (5,50 ), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        img_total = np.concatenate((image, res), axis = 1)
        print("img_total:", img_total.shape)
        # cv2.imshow('input img', img_total)
        # cv2.waitKey()
        video.write(img_total)

    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # videos_path = '/root/limiao/test.mp4'
    # frames_save_path = '/root/limiao/test11/'
    # time_interval = 50
    # video2frame(videos_path, frames_save_path, time_interval)

    # im_dir = '/root/limiao/test/'  # 帧存放路径
    # video_dir = '/root/limiao/res/test.mp4'  # 合成视频存放的路径
    # fps = 2
    # frame2video(im_dir, video_dir, fps)

    data_path1 = "/root/DeforumStableDiffusionLocal/output/2023-05/TaskName_extract1_075_alpha_invert/inputframes/"
    data_path2 = "/root/DeforumStableDiffusionLocal/output/2023-05/TaskName_extract1_075_alpha_invert/"
    merge_vide()