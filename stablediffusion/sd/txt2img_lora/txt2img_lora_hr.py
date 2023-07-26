#! /usr/bin/env python
# -*- coding: utf-8 -*-
import os
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
from diffusers import EulerAncestralDiscreteScheduler, AutoencoderKL, DDIMScheduler, DPMSolverSDEScheduler
from diffusers_017 import ControlNetModel
from diffusers_017 import  StableDiffusionPipeline, DiffusionPipeline
from controlnet_aux import CannyDetector, OpenposeDetector, MLSDdetector, HEDdetector, MidasDetector
import PIL.Image
import cv2, re
import numpy as np
from lora_model import Model

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
        if neg_seed.find(" --seed ") == -1:
            neg_prompt = neg_seed
            seed = None
        else:
            neg_prompt, seed = neg_seed.split(" --seed ")
        negative_prompt.append(neg_prompt)
        seeds.append(seed)
    prompts = prompt[-1:] * args.repeats
    negative_prompts = negative_prompt[-1:] * args.repeats
    seeds = seeds[-1:] * args.repeats
    print("prompts:", len(prompts), prompts)
    print("negative_prompts:", len(negative_prompts))
    print("seeds:", len(seeds), seeds)

    for i in range(0, len(prompts[:1])):
        # init generator according to fix_seeds
        # generator = torch.Generator(device='cuda').manual_seed(args.seed)
        generator = None

        if args.sampler == "euler_a":
            scheduler = EulerAncestralDiscreteScheduler.from_pretrained(args.model_path, subfolder="scheduler")
        elif args.sampler == "ddim":
            scheduler = DDIMScheduler.from_pretrained(args.model_path, subfolder="scheduler")
        elif args.sampler == "dpm_sde":
            scheduler = DPMSolverSDEScheduler.from_pretrained(args.model_path, subfolder="scheduler")
        pipe = StableDiffusionPipeline.from_pretrained(args.model_path,
                                                       scheduler=scheduler,
                                                       revision="fp16", torch_dtype=torch.float16)
        if args.vae_path is not None:
            vae = AutoencoderKL.from_pretrained(args.vae_path, subfolder="vae", torch_dtype=torch.float16).to("cuda")
            pipe.vae = vae
        # nsfw false
        pipe.safety_checker = lambda images, clip_input: (images, False)

        if args.lora_path is not None:
            Lora = Model()
            p = prompts[i]
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

        start = time.time()
        image = pipe(prompt=[prompts[i]]
                     ,negative_prompt=[negative_prompts[i]]
                     ,generator = generator
                     ,num_inference_steps = args.steps
                     ,guidance_scale = args.scale
                     ,seed = seeds[i]
                     ,num_images_per_prompt=args.num_images_per_prompt
                     ,height=args.H
                     ,width=args.W
                     ,Highres_strength=args.Highres_strength
                     ,Highres_scale=args.Highres_scale
                     ).images
        print("infer time: {}".format(time.time() - start))

        for j in range(0, len(image)):
            image[j].save(args.outdir + "/txt2img_{}_{}.png".format(str(i), str(j)))


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, default=None, help="if specified, load prompts from this file / 指定時はプロンプトをファイルから読み込む"
    )
    parser.add_argument(
        "--vae_path", type=str, default=None, help="if specified, load prompts from this file / 指定時はプロンプトをファイルから読み込む"
    )
    parser.add_argument(
        "--from_file", type=str, default=None, help="if specified, load prompts from this file / 指定時はプロンプトをファイルから読み込む"
    )
    parser.add_argument("--mask_path", type=str, default=None, help="mask in inpainting / inpaint時のマスク")
    parser.add_argument("--strength", type=float, default=1.0, help="img2img strength / img2img時のstrength")
    parser.add_argument("--scale",type=float,default=7.5, help="unconditional guidance scale: eps = eps(x, empty) +"
                                                               " scale * (eps(x, cond) - eps(x, empty)) / guidance scale",)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--W", type=int, default=512)
    parser.add_argument("--H", type=int, default=512)
    parser.add_argument("--seed", type=str, default=None)
    parser.add_argument("--repeats", type=int, default=1)  # total_img = len(prompt) * repeats * num_images_per_prompt
    parser.add_argument("--num_images_per_prompt", type=int, default=1)

    parser.add_argument("--outdir", type=str, default="outputs", help="dir to write results to / 生成画像の出力先")
    parser.add_argument("--lora_path", type=str, default=None,)
    parser.add_argument("--sampler", type=str, default='euler_a', choices=["euler_a", "dpm_sde", "ddim"])
    parser.add_argument("--Highres_strength", type=float, default=None)
    parser.add_argument("--Highres_scale", type=int, default=2)

    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    main(args)
