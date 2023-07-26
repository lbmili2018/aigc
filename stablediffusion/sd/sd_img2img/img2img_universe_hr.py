import random
import time
import torch
import sys
import argparse
for path in sys.path:
    print(path)
from torchvision import transforms
import requests
from PIL import Image
from io import BytesIO
from diffusers import EulerAncestralDiscreteScheduler, AutoencoderKL, StableDiffusionImg2ImgPipeline, StableDiffusionPipeline
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageOps, ImageFilter
import cv2, os, re
from lora_model import Model

device = "cuda"

def get_crop_region(mask, pad=0):
    """finds a rectangular region that contains all masked ares in an image. Returns (x1, y1, x2, y2) coordinates of the rectangle.
    For example, if a user has painted the top-right part of a 512x512 image", the result may be (256, 0, 512, 256)"""

    h, w = mask.shape

    crop_left = 0
    for i in range(w):
        if not (mask[:, i] == 0).all():
            break
        crop_left += 1

    crop_right = 0
    for i in reversed(range(w)):
        if not (mask[:, i] == 0).all():
            break
        crop_right += 1

    crop_top = 0
    for i in range(h):
        if not (mask[i] == 0).all():
            break
        crop_top += 1

    crop_bottom = 0
    for i in reversed(range(h)):
        if not (mask[i] == 0).all():
            break
        crop_bottom += 1

    return (
        int(max(crop_left - pad, 0)),
        int(max(crop_top - pad, 0)),
        int(min(w - crop_right + pad, w)),
        int(min(h - crop_bottom + pad, h))
    )



def expand_crop_region(crop_region, processing_width, processing_height, image_width, image_height):
    """expands crop region get_crop_region() to match the ratio of the image the region will processed in; returns expanded region
    for example, if user drew mask in a 128x32 region, and the dimensions for processing are 512x512, the region will be expanded to 128x128."""

    x1, y1, x2, y2 = crop_region

    ratio_crop_region = (x2 - x1) / (y2 - y1)
    ratio_processing = processing_width / processing_height

    if ratio_crop_region > ratio_processing:
        desired_height = (x2 - x1) / ratio_processing
        desired_height_diff = int(desired_height - (y2-y1))
        y1 -= desired_height_diff//2
        y2 += desired_height_diff - desired_height_diff//2
        if y2 >= image_height:
            diff = y2 - image_height
            y2 -= diff
            y1 -= diff
        if y1 < 0:
            y2 -= y1
            y1 -= y1
        if y2 >= image_height:
            y2 = image_height
    else:
        desired_width = (y2 - y1) * ratio_processing
        desired_width_diff = int(desired_width - (x2-x1))
        x1 -= desired_width_diff//2
        x2 += desired_width_diff - desired_width_diff//2
        if x2 >= image_width:
            diff = x2 - image_width
            x2 -= diff
            x1 -= diff
        if x1 < 0:
            x2 -= x1
            x1 -= x1
        if x2 >= image_width:
            x2 = image_width

    return x1, y1, x2, y2



def main(args):
    print(f"reading prompts from {args.from_file}")
    with open(args.from_file, "r", encoding="utf-8") as f:
        prompt_list = f.read().splitlines()
        prompt_list = [d for d in prompt_list if len(d.strip()) > 0]

    prompt, negative_prompt, seeds = [], [], []
    for all_prompt in prompt_list:
        pos_prompt, neg_seed = all_prompt.strip().split(" --neg ")
        prompt.append(pos_prompt)
        neg_prompt, seed = neg_seed.split(" --seed ")
        negative_prompt.append(neg_prompt)
        seeds.append(int(seed))

    prompts = prompt * args.repeats
    negative_prompts = negative_prompt* args.repeats
    seeds = seeds * args.repeats
    print("prompts:", len(prompts))
    print("negative_prompts:", len(negative_prompts))
    print("seeds:", len(seeds))

    idx = 1
    mask_ext=None
    for i in range(0, len(seeds), idx):
        print("seed[i:i+{}]:".format(idx), seeds[i:i+idx])

        scheduler = EulerAncestralDiscreteScheduler.from_pretrained(args.model_path, subfolder="scheduler")
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(args.model_path, scheduler=scheduler, revision="fp16", torch_dtype=torch.float16)
        if args.vae_path is not None:
            vae = AutoencoderKL.from_pretrained(args.vae_path, subfolder="vae", torch_dtype=torch.float16)
            pipe.vae = vae
        # nsfw false
        pipe.safety_checker = lambda images, clip_input: (images, False)

        if args.lora_path is not None:
            Lora = Model()
            p = prompts[0]
            p1 = re.compile(r'[<](.*?)[>]', re.S)  # 最小匹配
            Lora_info = re.findall(p1, p)
            print("Lora_info:", Lora_info)
            Lora_total_path = os.listdir(args.lora_path)
            for m, lora_info in enumerate(Lora_info):
                info = lora_info.split(":")
                weight = float(info[-1])
                lora_path = info[1]
                for k in range(len(Lora_total_path)):
                    if lora_path[:3] in Lora_total_path[k]:
                        lora_new = Lora_total_path[k]
                        break
                print("lora_name:", lora_new)
                Lora.set_lora(pipe, args.lora_path + lora_new, weight=weight)
        pipe.to(device)

        ### init generator according to fix_seeds
        # generator = torch.Generator(device='cuda').manual_seed(args.seed)
        generator = [torch.Generator(device='cuda').manual_seed(x) for x in seeds[i:i+idx]]
        # generator = None

        ### only support random_seed  for num_images  _per_prompt,
        ### To do fix_seed for num_images_per_prompt
        if args.num_images_per_prompt > 1:
            seeds = [random.randint(1, 1e7)  for i in range(len(seeds[i:i+idx]) * args.num_images_per_prompt)]
            generator = [torch.Generator(device='cuda').manual_seed(x) for x in seeds[i:i+idx]]

        # load image
        init_image = Image.open(args.image_path).convert("RGB").resize((args.W, args.H))
        orig_img = init_image
        # print("init_image:", np.array(init_image))
        new_init_image = []
        new_init_image.extend([init_image] * len(seeds[i:i+idx]))
        init_image = new_init_image

        if args.AD_Fix:
            print("starting ADetailer preprocessing")
            #### det face bbox, only single-person
            print("starting dect face")
            model = YOLO(args.face_model)
            output = model(args.image_path)
            print("pred:", output[0].boxes.shape)
            bbox = output[0].boxes.cpu()
            bboxes = bbox.boxes # dect all bboxes
            print("bboxes:,", bboxes)
            # only select first bbox
            x1 = int(bboxes[0, 0])
            y1 = int(bboxes[0, 1])
            x2 = int(bboxes[0, 2])
            y2 = int(bboxes[0, 3])
            print("orig x1, y1, x2, y2,", x1, y1, x2, y2)
            face = init_image[0].crop((x1, y1, x2, y2))
            face.save("./face.png")
            new_mask = np.zeros_like(init_image[0])
            new_mask[int(y1):int(y2), int(x1):int(x2), ...] = 255
            # new_mask = cv2.bitwise_not(new_mask)
            new_mask = Image.fromarray(new_mask.astype('uint8')).convert('L')
            print("ending dect face")

            mask = new_mask.filter(ImageFilter.GaussianBlur(4))
            mask_w, mask_h = mask.size
            crop_region = get_crop_region(np.array(new_mask), pad=32)
            crop_region = expand_crop_region(crop_region, args.W, args.H, mask_w, mask_h)
            x1, y1, x2, y2 = crop_region
            print("extension x1, y1, x2, y2:", x1, y1, x2, y2)
            mask_ext = mask.crop(crop_region)
            mask_ext_w, mask_ext_h = mask_ext.size
            mask_ext = mask_ext.resize((args.W, args.H))

            face_ext = init_image[0].crop(crop_region)
            face_ext = face_ext.resize((args.W, args.H))
            mask_ext.save("./inp_mask_ext.png")
            face_ext.save("./inp_face_ext.png")
            init_image = [face_ext]
            print("ending ADetailer preprocessing")
            ##################

        start = time.time()
        image = pipe(prompt=prompts[i:i+idx]
                     ,negative_prompt=negative_prompts[i:i+idx]
                     ,image=init_image
                     ,mask_image=mask_ext
                     ,generator = generator
                     ,num_inference_steps = args.steps
                     ,strength = args.strength
                     ,guidance_scale = args.scale
                     ,seed = seeds[i:i+idx]
                     ,num_images_per_prompt=args.num_images_per_prompt
                     ,AD_Fix=args.AD_Fix
                     ).images
        print("infer time: {}".format(time.time() - start))

        if args.AD_Fix:
            print("starting ADetailer post-processing")
            for j in range(0, len(image)):
                image[0].save('./res_sd.png')
                face_res = image[j].resize((mask_ext_w,mask_ext_h))
                face_res = np.array(face_res)
                orig_img = np.array(orig_img)
                orig_img[y1:y2, x1:x2, ...] = face_res
                final = Image.fromarray(orig_img)
                final.save(args.outdir + "/img_{}_{}_{}.png".format(str(i), str(j), str(seeds[i:i+idx][j])))
                print("ending ADetailer post-processing")
        else:
            for j in range(0, len(image)):
                image[j].save(args.outdir + "/img_{}_{}_{}.png".format(str(i), str(j), str(seeds[i:i+idx][j])))


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, default=None, help="if specified, load prompts from this file / 指定時はプロンプトをファイルから読み込む"
    )
    parser.add_argument(
        "--vae_path", type=str, default=None, help="if specified, load prompts from this file / 指定時はプロンプトをファイルから読み込む"
    )
    parser.add_argument(
        "--image_path", type=str, default=None, help="image to inpaint or to generate from / img2imgまたはinpaintを行う元画像"
    )
    parser.add_argument(
        "--from_file", type=str, default=None, help="if specified, load prompts from this file / 指定時はプロンプトをファイルから読み込む"
    )
    parser.add_argument("--mask_path", type=str, default=None, help="mask in inpainting / inpaint時のマスク")
    parser.add_argument("--strength", type=float, default=None, help="img2img strength / img2img時のstrength")
    parser.add_argument("--scale",type=float,default=7.5, help="unconditional guidance scale: eps = eps(x, empty) +"
                                                               " scale * (eps(x, cond) - eps(x, empty)) / guidance scale",)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--W", type=int, default=512)
    parser.add_argument("--H", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--repeats", type=int, default=1)  # total_img = len(prompt) * repeats * num_images_per_prompt
    parser.add_argument("--num_images_per_prompt", type=int, default=1)
    parser.add_argument("--outdir", type=str, default="outputs", help="dir to write results to / 生成画像の出力先")
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--face_model", type=str, default=None)
    parser.add_argument("--AD_Fix", type=str, default=False)


    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    main(args)
