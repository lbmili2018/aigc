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
from diffusers_017 import StableDiffusionControlNetPipeline
from controlnet_aux import CannyDetector, OpenposeDetector, MLSDdetector, HEDdetector, MidasDetector
import PIL.Image
import cv2
import  re
import  os
import numpy as np
from lora_model import Model
from transformers import pipeline
from diffusers.utils import load_image

device = "cuda"
canny_low_thr = None
canny_high_thr = None

class MidasDetectorWrapper:
	''' a wrapper around the midas detector model which allows
	choosing either the depth or the normal output on creation '''
	def __init__(self, output_index=0, **kwargs):
		self.model = pipeline('depth-estimation') #MidasDetector()
		self.output_index = output_index
		self.default_kwargs = dict(kwargs)
	def __call__(self, image, **kwargs):
		ka = dict(list(self.default_kwargs.items()) + list(kwargs.items()))
		#return torch.tensor(self.model(np.asarray(image), **ka)[self.output_index][None, :, :].repeat(3,0)).unsqueeze(0)
		return PIL.Image.fromarray(self.model(np.asarray(image), **ka)[self.output_index]).convert("RGB")


def main(args):
    print("args.controlnet:", args.controlnet)
    control_name = [x for x in args.controlnet.split(" ")]
    control_scale = [float(x) for x in args.control_scale.split(" ")]
    print("control_name:", control_name)
    print("control_scale:", control_scale)
    controlnets = []
    detectors = dict()
    if 'canny' in control_name:
        detector_kwargs = dict({
            "low_threshold": canny_low_thr if canny_low_thr != None else 50,
            "high_threshold": canny_high_thr if canny_high_thr != None else 200
        })
        controlnet_model = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
        detectors['canny'] = CannyDetector()
        controlnets.append(controlnet_model)
    if 'depth' in control_name:
        detector_kwargs = dict()
        # detector_model = MidasDetectorWrapper()
        controlnet_model = ControlNetModel.from_pretrained("/dnwx/datasets/lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16)
        detectors['depth'] = pipeline('depth-estimation')
        controlnets.append(controlnet_model)

    if 'hed' in control_name:
        detector_kwargs = dict()
        controlnet_model = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-hed", torch_dtype=torch.float16)
        detectors['hed'] = HEDdetector.from_pretrained("lllyasviel/ControlNet")
        controlnets.append(controlnet_model)
    # print("controlnet_model:", controlnet_model)

    if args.sampler == "euler_a":
        scheduler = EulerAncestralDiscreteScheduler.from_pretrained(args.model_path, subfolder="scheduler")
    elif args.sampler == "ddim":
        scheduler = DDIMScheduler.from_pretrained(args.model_path, subfolder="scheduler")
    elif args.sampler == "dpm_sde":
        scheduler = DPMSolverSDEScheduler.from_pretrained(args.model_path, subfolder="scheduler")

    pipe = StableDiffusionControlNetPipeline.from_pretrained(args.model_path,
                                                                    controlnet=controlnets,
                                                                    scheduler=scheduler,
                                                                    revision="fp16", torch_dtype=torch.float16)

    if args.vae_path is not None:
        vae = AutoencoderKL.from_pretrained(args.vae_path, subfolder="vae", torch_dtype=torch.float16).to("cuda")
        pipe.vae = vae

    #nsfw false
    pipe.safety_checker = lambda images, clip_input: (images, False)


    print(f"reading prompts from {args.from_file}")
    with open(args.from_file, "r", encoding="utf-8") as f:
        prompt_list = f.read().splitlines()
        prompt_list = [d for d in prompt_list if len(d.strip()) > 0]
    prompt, negative_prompt, seeds, img_paths = [], [], [], []
    for all_prompt in prompt_list:
        pos_prompt, neg_seed = all_prompt.strip().split(" --neg ")
        prompt.append(pos_prompt)
        if neg_seed.find(" --seed ") == -1:
            neg_prompt, img_path = neg_seed.split(" --img_path ")
            seed = None
        else:
            neg_prompt, seed_imgpath = neg_seed.split(" --seed ")
            seed, img_path = seed_imgpath.split(" --img_path ")
        negative_prompt.append(neg_prompt)
        seeds.append(seed)
        img_paths.append(img_path)

    prompts = prompt * args.repeats
    negative_prompts = negative_prompt * args.repeats
    seeds = seeds * args.repeats
    print("prompts:", len(prompts), prompts)
    print("negative_prompts:", len(negative_prompts))
    print("seeds:", len(seeds), seeds)


    for i in range(0, len(prompts)):
        generator = None
        if args.num_images_per_prompt > 1:
            # seeds = [random.randint(1, 1e7)  for i in range(len(seeds[i:i+idx]) * args.num_images_per_prompt)]
            # generator = [torch.Generator(device='cuda').manual_seed(x) for x in seeds[i:i+idx]]
            generator = None
        # load image
        init_image = Image.open(img_paths[i]).convert("RGB").resize((args.W, args.H))
        print("init_image:", init_image.size)

        control_image = []
        for key, vaule in detectors.items():
            if key == "depth":
                image = detectors[key](init_image)['depth']
                image = np.array(image)
                image = image[:, :, None]
                image = np.concatenate([image, image, image], axis=2)
                ss = Image.fromarray(image)
            else:
                ss = detectors[key](init_image, **detector_kwargs).convert("RGB")
            ss = ss.resize((args.W, args.H))
            ss.save(args.outdir + "/{}_{}_{}.png".format(key, args.W, args.H))
            control_image.append(ss)

        if args.lora_path is not None:
            Lora = Model()
            str = prompts[i]
            p1 = re.compile(r'[<](.*?)[>]', re.S)  # 最小匹配
            Lora_info = re.findall(p1, str)
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
                print("lora_new:", lora_new)
                Lora.set_lora(pipe, args.lora_path + lora_new, weight=weight)
        pipe.to(device)

        print("prompts:", prompts[i])
        start = time.time()
        image = pipe(prompt=[prompts[i]]
                     ,image= control_image
                     ,negative_prompt=[negative_prompts[i]]
                     ,generator = generator
                     ,controlnet_conditioning_scale=control_scale
                     ,num_inference_steps = args.steps
                     ,guidance_scale = args.scale
                     ,seed = seeds[i]
                     ,num_images_per_prompt=args.num_images_per_prompt
                     ).images
        print("infer time: {}".format(time.time() - start))

        for j in range(0, len(image)):
            image[j].save(args.outdir + "/img_{}.png".format(j))


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
    parser.add_argument("--controlnet", type=str, default=None)
    parser.add_argument("--control_scale", type=str, default=None)


    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    main(args)
