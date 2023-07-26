import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
from PIL import Image
from io import BytesIO
import time
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionControlNetPipeline, ControlNetModel,AutoencoderKL
from diffusers import UniPCMultistepScheduler, EulerAncestralDiscreteScheduler,\
	DPMSolverMultistepScheduler, DPMSolverSinglestepScheduler

torch.backends.cuda.matmul.allow_tf32 = True

# model_id_or_path = "./checkpoints/toonyou_beta1.safetensors"
# pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path
# 	, torch_dtype=torch.float16)
# pipe = pipe.to("cuda")
# pipe.enable_xformers_memory_efficient_attention()

# pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
#     "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
# )

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "/dnwx/datasets/models/Stable-diffusion/diff", torch_dtype=torch.float16
)
vae = AutoencoderKL.from_pretrained("/dnwx/datasets/models/Stable-diffusion/vae", torch_dtype=torch.float16)
pipe.vae = vae
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
pipe.enable_xformers_memory_efficient_attention()

image_path = "/root/limiao/orig.jpg"
init_image = Image.open(image_path).convert("RGB")
init_image = init_image.resize((540, 960))


prompt = ["""(Angel with open wings:1.5), (white blond hair:1.2) , tarot, vulcan salute, Attention, halo behind head, finely_detailed,perfect_eyes,perfect_face, perfect_fingers, Golden rose ,(crown:1.2), golden light, shine""",
 ]
negative_prompt = ["""(nsfw:2.0), (nude:2.0), (Nipple:2.0), (Cleavage:0.8), (Big breasts:1.5),(Detailed of man:2.0), (Masculine characteristics:2.0),(Masculinity:2.0),(Male:2.0),(Cross-eyed:2.0),(Beard:1.3),(beard stubble:1.5),((text)), (((title))), ((logo)),(worst quality:2.0), (Nevus:2.0), (age spot:2.0), (Acne:2.0),(skin spots:2.0), ((Black patches appear on the face)), (disfigured:1.2), (low quality:2.0), (low quality:2), (deformed),  (deformed, distorted, disfigured:1.3), (duplicate:2.0), distorted, (((tranny))), (unclear eyes:1.331), poorly drawn, bad anatomy, wrong anatomy, full Body, extra limb, skin blemishes, missing limb, floating limbs, (mutated hands and fingers:1.4), Face occlusion, ((Multiple people)), disconnected limbs, extra limbs, mutation, (bad proportions:1.3), mutated, ugly, disgusting, amputation, duplicate, bugly, huge eyes, (mutilated:1.2), ((worst face)), (bad and mutated hands:1.3), horror, geometry, bad_prompt, (bad hands), (((missing fingers))), multiple limbs, bad anatomy, ((morbid)), (interlocked fingers:1.2), (((Ugly Fingers))), (extra digit and hands and fingers and legs and arms:1.4), (deformed fingers:1.2), (long fingers:1.4), (bad-artist-anime), bad-artist, bad hand""",
]

generator = torch.Generator("cuda").manual_seed(4154105873)
a = time.time()
images = pipe(prompt=prompt*5
	, negative_prompt = negative_prompt*5
	, image=init_image
	, num_inference_steps=20
	, guidance_scale=7
	, strength=0.5
	, generator=generator).images

# generator = torch.manual_seed(2)
# output = pipe(
#     prompt*4,
#     init_image,
#     negative_prompt=negative_prompt*4,
#     num_inference_steps=20,
#     generator=generator,
# )
print("出图耗时: {}".format(time.time()-a))


for i in range(0, len(images)):
	images[i].save("/root/limiao/img2img_fantasy_landscape_{}.png".format(i))
