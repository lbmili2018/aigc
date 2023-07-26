import torch
import os, argparse
from PIL import Image
import numpy as np
from controlnet_aux import PidiNetDetector, HEDdetector, CannyDetector

from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
    AutoencoderKL,
    EulerAncestralDiscreteScheduler,
)
torch.backends.cuda.matmul.allow_tf32 = True

def main(args):
    controlnet = ControlNetModel.from_pretrained(args.ctl_path, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(args.sd_path, controlnet=controlnet, torch_dtype=torch.float16)
    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)
    if args.vae_path is not None:
        vae = AutoencoderKL.from_pretrained(args.vae_path, subfolder="vae", torch_dtype=torch.float16).to("cuda")
        pipe.vae = vae

    image = Image.open(args.image_path).convert("RGB")
    orig_W, orig_H = image.size
    print("image orig size:", image.size)

    image = image.resize((args.W, args.H))#.convert("1")
    prompt = " "
    negative_prompt = 'nsfw, nude, worst quality, Acne, worst face, disfigured, duplicate, Multiple people,poorly drawn, wrong anatomy, bad proportions, full body, extra limb, skin blemishes, missing limb, deformed fingers, mutated hands and fingers, disconnected limbs,ugly,mutilated, bad and mutated hands, multiple limbs'
    if args.from_file is not None:
        print(f"reading prompts from {args.from_file}")
        with open(args.from_file, "r", encoding="utf-8") as f:
            prompt_list = f.read().splitlines()
            prompt_list = [d for d in prompt_list if len(d.strip()) > 0]
        prompt =prompt_list[0]
    print("prompt:", prompt)
    if args.is_hed:
        hed = HEDdetector.from_pretrained('lllyasviel/ControlNet') # Annotators, ControlNet
        image = hed(image, scribble=True)
    elif args.is_canny:
        canny = CannyDetector()
        image = canny(image)
    else:
        image = image.convert("1")
    
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    if args.seed is not None:
        generator = torch.Generator(device='cuda').manual_seed(args.seed)
        print("fixed seed:", args.seed)
    else:
        generator = None
        print("random seed")

    image.resize((orig_W, orig_H)).save('{}/{}_scribble.jpg'.format(args.outdir, args.image_path.split("/")[-1].split(".")[0]))
    image = pipe(prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                num_inference_steps=args.steps,
                generator=generator,
                num_images_per_prompt=args.num_images_per_prompt,
                ).images

    res = []
    for i in range(0, len(image)):
        r, g, b = image[i].getextrema()
        if r[1] == 0 and g[1] == 0 and b[1] == 0:
            # print("black img")
            continue
        res.append(image[i])

    final = res[0].resize((orig_W, orig_H))
    print("image final size:", final.size)
    final.save('{}/final_out.jpg'.format(args.outdir)) #str(i).zfill(6)


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sd_path", type=str, default=None)
    parser.add_argument("--ctl_path", type=str, default=None)
    parser.add_argument("--vae_path", type=str, default=None)
    parser.add_argument("--is_hed", type=bool, default=False)
    parser.add_argument("--is_canny", type=bool, default=False)
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--from_file", type=str, default=None, help="if specified, load prompts from this file")
    parser.add_argument("--scale",type=float,default=7, help="unconditional guidance scale: eps = eps(x, empty) +"
                                                               " scale * (eps(x, cond) - eps(x, empty)) / guidance scale",)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--W", type=int, default=512)
    parser.add_argument("--H", type=int, default=640)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=1)  # total_img = len(prompt) * repeats * num_images_per_prompt
    parser.add_argument("--num_images_per_prompt", type=int, default=5)
    parser.add_argument("--outdir", type=str, default="outputs", help="dir to write results to / 生成画像の出力先")

    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    main(args)
