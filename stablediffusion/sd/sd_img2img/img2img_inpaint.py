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
from diffusers import EulerAncestralDiscreteScheduler, AutoencoderKL ,StableDiffusionInpaintPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionControlNetInpaintPipeline
import numpy as np
from ultralytics import YOLO
import cv2, os, re
from lora_model import Model

def numpy_to_pt(images):
    images = torch.from_numpy(images.transpose(0, 3, 1, 2))
    return images

def pt_to_numpy(images):
    images = images.float().numpy()
    return images

device = "cuda"
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
    for i in range(0, len(seeds), idx):

        scheduler = EulerAncestralDiscreteScheduler.from_pretrained(args.model_path, subfolder="scheduler")
        pipe = StableDiffusionInpaintPipeline.from_pretrained(args.model_path, scheduler=scheduler, revision="fp16", torch_dtype=torch.float16)
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

        # init generator according to fix_seeds
        # generator = torch.Generator(device='cuda').manual_seed(args.seed)
        generator = [torch.Generator(device='cuda').manual_seed(x) for x in seeds[i:i+idx]]

        # only support random_seed  for num_images  _per_prompt,
        # To do fix_seed for num_images_per_prompt
        if args.num_images_per_prompt > 1:
            seeds = [random.randint(1, 1e7)  for i in range(len(seeds[i:i+idx]) * args.num_images_per_prompt)]
            generator = [torch.Generator(device='cuda').manual_seed(x) for x in seeds[i:i+idx]]

        # load image
        init_image = Image.open(args.image_path).convert("RGB").resize((args.W, args.H))
        # print("init_image:", np.array(init_image))
        new_init_image = []
        new_init_image.extend([init_image] * len(seeds[i:i+idx]))
        init_image = new_init_image

        mask_image = args.image_path
        path = "./face_yolov8n_v2.pt"
        model = YOLO(path)
        output = model(mask_image)
        print("output:", output[0].boxes.shape)
        bbox = output[0].boxes.cpu()
        # mask = output[0]
        # print("mask:", mask)
        bbox11 = bbox.boxes
        print("bbox11:,", bbox11)
        x1 = float(bbox11[0, 0])
        y1 = float(bbox11[0, 1])
        x2 = float(bbox11[0, 2])
        y2 = float(bbox11[0, 3])
        print("x1, y1, x2, y2,", x1, y1, x2, y2)
        face = init_image[0].crop((x1, y1, x2, y2))
        face.save("./txt2img_0.png")

        # mask_image = Image.open("./txt2img_0.png").convert("RGB").resize((args.W, args.H))
        # mask_image = np.array(mask_image)
        new_mask = np.zeros_like(init_image[0])
        new_mask[int(y1):int(y2), int(x1):int(x2),...] = 255
        new_mask = Image.fromarray(new_mask.astype('uint8')).convert('L')
        new_mask.save("./txt2img_new_mask.png")


        # def download_image(url):
        #     response = requests.get(url)
        #     return Image.open(BytesIO(response.content)).convert("RGB")
        # img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
        # mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"
        #
        # init_image = download_image(img_url).resize((512, 512))
        # mask_image = download_image(mask_url).resize((512, 512))


        start = time.time()
        image = pipe(prompt=prompts[i:i+idx]
                     ,negative_prompt=negative_prompts[i:i+idx]
                     ,image=init_image
                     ,mask_image=new_mask
                     ,height=args.H
                     ,width=args.W
                     ,generator = generator
                     ,num_inference_steps = args.steps
                     ,strength = args.strength
                     ,guidance_scale = args.scale
                     # ,seed = seeds[i:i+idx]
                     ,num_images_per_prompt=args.num_images_per_prompt
                     ).images
        print("infer time: {}".format(time.time() - start))

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
    parser.add_argument("--W", type=int, default=720)
    parser.add_argument("--H", type=int, default=960)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--repeats", type=int, default=1)  # total_img = len(prompt) * repeats * num_images_per_prompt
    parser.add_argument("--num_images_per_prompt", type=int, default=1)
    parser.add_argument("--outdir", type=str, default="outputs", help="dir to write results to / 生成画像の出力先")
    parser.add_argument("--lora_path", type=str, default=None,)

    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    main(args)
