import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import library.model_util as model_util
import torch

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
        os.system("python gen_img_diffusers_deploy.py "
                  "--ckpt /root/algo/stable-diffusion-webui/models/Stable-diffusion/WAVE-P.ckpt "
                  "--vae /dnwx/datasets/models/VAE/vae-ft-mse-840000-ema-pruned/vae-ft-mse-840000-ema-pruned.ckpt "
                  "--from_file ./prompts.txt "
                  "--batch_size 1 --images_per_prompt 1 --sequential_file_name "
                  "--seed 1 "
                  "--diffusers_xformers --xformers --fp16 "
                  "--image_path {} --strength {} "
                  " --W 512 --H 512 --steps 20 --scale 7 --sampler euler_a "
                  "--outdir /root/limiao/result/res_universe".format(item.img_path, item.strength)
                  )
        print("ai universe is completed")
        return "result save path: /root/limiao/result/res_universe"
    except Exception as e:
        return  e

if __name__ == '__main__':
    # text_encoder, vae, unet = model_util.load_models_from_stable_diffusion_checkpoint(v2=False, ckpt_path=sd_model_path)
    # vae = model_util.load_vae(new_vae_path, dtype=torch.float16)
    uvicorn.run(app=app,
                host="0.0.0.0",
                port=8081,
                workers=1)


# #定义变量
# para1=''
# para2=''
# #调用py脚本，并传递参数
# python test.py $para1 $para2