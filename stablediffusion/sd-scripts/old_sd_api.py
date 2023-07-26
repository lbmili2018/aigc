
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import library.model_util as model_util
import torch
import gen_img_diffusers_deploy

class Item(BaseModel):
    img_path: str
    strength: float = 0.4

app = FastAPI()

@app.get("/")
async  def root():
    return 'Hello World!'


@app.post("/ai_universe")
async def fcao_predict(item: Item):
    # os.system("./ai_universe.sh")
    if not os.path.exists(item.img_path):
        raise HTTPException(status_code=404, detail="image path not found")
    try:
        parser = gen_img_diffusers_deploy.setup_parser()
        args = parser.parse_args()
        args.batch_size = 1
        args.images_per_prompt = 1
        args.sequential_file_name = True
        args.seed = 4154105873
        args.diffusers_xformers = True
        args.xformers = True
        args.fp16 = True

        args.W = 512
        args.H = 960
        args.steps = 20
        args.scale = 8
        args.sampler = "euler_a"
        args.outdir = "/root/limiao/result/res_universe"
        args.from_file = "./prompts_test.txt"

        args.image_path = item.img_path
        args.strength = item.strength

        print("args", args)
        gen_img_diffusers_deploy.run(args, text_encoder, vae, unet)
        print("ai universe is completed")
        return "result save path: /root/limiao/result/res_universe"
    except Exception as e:
        return  e

if __name__ == '__main__':
    sd_model_path = "/root/algo/stable-diffusion-webui/models/Stable-diffusion/ghostmix_v12.safetensors"
    new_vae_path = "/dnwx/datasets/models/VAE/vae-ft-mse-840000-ema-pruned/vae-ft-mse-840000-ema-pruned.ckpt"
    print("sd_model:", sd_model_path)
    text_encoder, vae, unet = model_util.load_models_from_stable_diffusion_checkpoint(v2=False, ckpt_path=sd_model_path)
    vae = model_util.load_vae(new_vae_path, dtype=torch.float16)
    uvicorn.run(app=app,
                host="0.0.0.0",
                port=8081,
                workers=1)


# #定义变量
# para1=''
# para2=''
# #调用py脚本，并传递参数
# python test.py $para1 $para2