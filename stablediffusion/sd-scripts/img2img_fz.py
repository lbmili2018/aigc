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
from diffusers import StableUnCLIPImg2ImgPipeline, EulerAncestralDiscreteScheduler, AutoencoderKL, StableDiffusionImg2ImgPipeline

device = "cuda"
def main(args):
    scheduler = EulerAncestralDiscreteScheduler.from_pretrained(args.model_path, subfolder="scheduler")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(args.model_path, scheduler=scheduler, revision="fp16", torch_dtype=torch.float16).to(device)

    if args.vae_path is not None:
        vae = AutoencoderKL.from_pretrained(args.vae_path, subfolder="vae", torch_dtype=torch.float16).to("cuda")
        pipe.vae = vae

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
    prompts = prompt *5
    negative_prompts = negative_prompt*5
    seeds = seeds * 5
    print("prompts:", len(prompts))
    print("negative_promptnegative_prompts:", len(negative_prompts))
    print("seeds:", len(seeds))


    # init generator according to seeds
    # generator = torch.Generator(device='cuda').manual_seed(args.seed)
    generator = [torch.Generator(device='cuda').manual_seed(x) for x in seeds]
    # load image
    init_image = Image.open(args.image_path).convert("RGB").resize((args.W, args.H))
    new_init_image = []
    new_init_image.extend([init_image] * len(seeds))
    init_image = new_init_image
    print("init_image:", len(init_image))

    start = time.time()
    image = pipe(prompt=prompts
                 ,negative_prompt=negative_prompts
                 ,image=init_image
                 ,generator = generator
                 ,num_inference_steps = args.steps
                 ,strength = args.strength
                 ,guidance_scale = args.scale
                 ,seed = seeds
                 ).images
    print("infer time: {}".format(time.time() - start))
    for i in range(0, len(image)):
        image[i].save(args.outdir + "/gen_img_{}.png".format(str(i).zfill(6)))


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
    parser.add_argument("--outdir", type=str, default="outputs", help="dir to write results to / 生成画像の出力先")

    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    main(args)
