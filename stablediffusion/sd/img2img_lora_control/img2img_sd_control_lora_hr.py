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
from diffusers import EulerAncestralDiscreteScheduler, AutoencoderKL, DDIMScheduler
from diffusers_017 import ControlNetModel
from diffusers_017 import StableDiffusionControlNetImg2ImgPipeline, StableDiffusionControlNetPipeline
from stable_diffusion_controlnet_img2img import StableDiffusionControlNetImg2ImgPipeline

from controlnet_aux import CannyDetector, OpenposeDetector, MLSDdetector, HEDdetector, MidasDetector
import PIL.Image
import cv2
import numpy as np
from lora_model import Model

device = "cuda"
canny_low_thr = None
canny_high_thr = None

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


def main(args):
    if args.controlnet == 'canny':
        detector_kwargs = dict({
            "low_threshold": canny_low_thr if canny_low_thr != None else 50,
            "high_threshold": canny_high_thr if canny_high_thr != None else 200
        })
        detector_model = CannyDetector()
        controlnet_model = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
    elif args.controlnet == 'depth':
        detector_kwargs = dict()
        detector_model = MidasDetectorWrapper()
        controlnet_model = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16)
    elif args.controlnet == 'hed':
        detector_kwargs = dict()
        detector_model = HEDdetector.from_pretrained("lllyasviel/ControlNet")
        controlnet_model = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-hed", torch_dtype=torch.float16)

    scheduler = EulerAncestralDiscreteScheduler.from_pretrained(args.model_path, subfolder="scheduler")
    print("controlnet_model:", controlnet_model)
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(args.model_path,
                                                                    controlnet=controlnet_model,
                                                                    scheduler=scheduler,
                                                                    revision="fp16", torch_dtype=torch.float16)

    if args.vae_path is not None:
        vae = AutoencoderKL.from_pretrained(args.vae_path, subfolder="vae", torch_dtype=torch.float16).to("cuda")
        pipe.vae = vae

    if args.lora_path is not None:
        Lora = Model()
        Lora.set_lora(pipe, args.lora_path)
    pipe.to(device)
    #nsfw false
    pipe.safety_checker = lambda images, clip_input: (images, False)


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
        print("seed[i:i+{}]:".format(idx), seeds[i:i+idx])
        # init generator according to fix_seeds
        # generator = torch.Generator(device='cuda').manual_seed(args.seed)
        generator = [torch.Generator(device='cuda').manual_seed(x) for x in seeds[i:i+idx]]

        # only support random_seed  for num_images_per_prompt,
        # To do fix_seed for num_images_per_prompt
        if args.num_images_per_prompt > 1:
            seeds = [random.randint(1, 1e7)  for i in range(len(seeds[i:i+idx]) * args.num_images_per_prompt)]
            generator = [torch.Generator(device='cuda').manual_seed(x) for x in seeds[i:i+idx]]
            # seeds = [random.randint(1, 1e7)  for i in range(len(seeds) * args.num_images_per_prompt)]
            # generator = [torch.Generator(device='cuda').manual_seed(x) for x in seeds]

        # load image
        init_image = Image.open(args.image_path).convert("RGB").resize((args.W, args.H))
        print("init_image:", init_image.size)

        new_init_image = []
        new_init_image.extend([init_image] * len(seeds[i:i+idx]))
        init_image = new_init_image

        control_image = []
        for xx in init_image:
            ss = detector_model(xx, **detector_kwargs).convert("RGB").resize((args.W, args.H))
            print("ss:", ss.size)
            control_image.append(ss)

        control_image[0].save(args.outdir + "/{}_{}.png".format(args.controlnet, 512_512))

        start = time.time()
        image = pipe(prompt=prompts[i:i+idx]
                     ,negative_prompt=negative_prompts[i:i+idx]
                     ,image=init_image
                     ,generator = generator
                     ,controlnet_conditioning_image = control_image # controlnet_conditioning_image, control_image
                     ,num_inference_steps = args.steps
                     ,strength = args.strength
                     ,guidance_scale = args.scale
                     ,seed = seeds[i:i+idx]
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
    parser.add_argument("--strength", type=float, default=1.0, help="img2img strength / img2img時のstrength")
    parser.add_argument("--scale",type=float,default=7.5, help="unconditional guidance scale: eps = eps(x, empty) +"
                                                               " scale * (eps(x, cond) - eps(x, empty)) / guidance scale",)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--W", type=int, default=512)
    parser.add_argument("--H", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--repeats", type=int, default=1)  # total_img = len(prompt) * repeats * num_images_per_prompt
    parser.add_argument("--num_images_per_prompt", type=int, default=1)
    parser.add_argument("--outdir", type=str, default="outputs", help="dir to write results to / 生成画像の出力先")
    parser.add_argument("--controlnet", type=str, default='hed', choices=["hed", "canny", "depth"])
    parser.add_argument("--lora_path", type=str, default=None,)


    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    main(args)
