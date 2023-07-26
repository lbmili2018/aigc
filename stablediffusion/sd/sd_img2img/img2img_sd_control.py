import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
from PIL import Image
from io import BytesIO
import time
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import UniPCMultistepScheduler

torch.backends.cuda.matmul.allow_tf32 = True


# model_id_or_path = "./checkpoints/toonyou_beta1.safetensors"
# pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path
# 	, torch_dtype=torch.float16)
# pipe = pipe.to("cuda")
# pipe.enable_xformers_memory_efficient_attention()

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
pipe.enable_xformers_memory_efficient_attention()



image_path = "/root/limiao/girl.jpg"
init_image = Image.open(image_path).convert("RGB")
init_image = init_image.resize((512, 512))

prompt = ["""Colorful hair,Detailed Eyes,shiny clothes,trend wear,Facial features details, colorful details, (face focus:1.5), wearing frosted jackets,
(armoured), (Machine armour), biomechanical cyborg, mecha musume, robot joints, (cool mechanical), (mecha:1.2), (anatomical), (robotic arms), (metal armor), ((Metal machinery)), ((precision mechanical parts)), microchip,
(technological sence), (cyborg:1.5),iridescent colors, futuristic room, Background virtualcolorful details,
(intricate details), (ultra detailed), CG, octane render, complex 3d render, holographic, (cyberpunk:1.3), cinematic shot, 
vignette, Edge light, fine luster, volumetric lighting, fluorescence, beautiful natural soft rim light, bright light,  power of the god""",
 """Spirit, flowers in clusters, Transparent wings,
 [Pink|Light yellow|Pink-orange|Light green|Light blue|Lavender] Flowers in the background
 ,[[Background virtual]],Forest,Hand bouquet, hair bow, hair ornament
 , hair ornament, hair bow, jewelry, flower on head, head wreath
 , (pointy ears:1.3), crystal earrings, crystal earrings, light smile, happy
 , --version niji, HD --style, 8k --hd, god rays, vignetting, bloom, Surrealism
 , rococo style, Romanticism, Baroque, Luminism, Surrealism, UHD, anatomically correct
 , textured skin, super detail, high details, best quality, high quality, 1080P, retina, 16k,Upright view""",
 """colorful details, (intricate details), (ultra detailed)
 , volumetric lighting, vignette, beautiful natural soft rim light
 , bright light, fluorescence, transparent, Background virtualcolorful details
 , iridescent colors, power of the god, octane render""",
 ]
negative_prompt = ["""(nsfw),(nude),(Nipple),(Cleavage:0.5),(Big breasts:1.5),
(worst quality:2.0),(Nevus:1.5), age spot, Acne, ((Black patches appear on the face)), skin spots, (disfigured:1.2), (low quality:2.0), (low quality:2), (deformed), ((text)), (((title))), ((logo)), (deformed, distorted, disfigured:1.3), (duplicate:2.0), distorted, (((tranny))), (unclear eyes:1.331), poorly drawn, bad anatomy, wrong anatomy, full Body, extra limb, skin blemishes, missing limb, floating limbs, (mutated hands and fingers:1.4), Face occlusion, ((Multiple people)), disconnected limbs, extra limbs, mutation, (bad proportions:1.3), mutated, ugly, disgusting, amputation, duplicate, bugly, huge eyes, monochrome, (mutilated:1.2), ((worst face)), (bad and mutated hands:1.3), horror, geometry, bad_prompt, (bad hands), (((missing fingers))), multiple limbs, bad anatomy, ((morbid)), (interlocked fingers:1.2), (((Ugly Fingers))), (extra digit and hands and fingers and legs and arms:1.4), (deformed fingers:1.2), (long fingers:1.2), (bad-artist-anime), bad-artist, bad hand""",
"""(nsfw),(nude),(Nipple:2), (Cleavage:0.5),(text:2.0),(logo:2.0),(title:2.0),(worst quality:2.0), (Nevus:1.5), age spot, Acne, ((Black patches appear on the face)), skin spots, (disfigured:1.2), (low quality:2.0), (low quality:2), (deformed), ((text)), (((title))), ((logo)), (deformed, distorted, disfigured:1.3), (duplicate:2.0), distorted, (((tranny))), (unclear eyes:1.331), poorly drawn, bad anatomy, wrong anatomy, full Body, extra limb, skin blemishes, missing limb, floating limbs, (mutated hands and fingers:1.4), Face occlusion, ((Multiple people)), disconnected limbs, extra limbs, mutation, (bad proportions:1.3), mutated, ugly, disgusting, amputation, duplicate, bugly, huge eyes, monochrome, (mutilated:1.2), ((worst face)), (bad and mutated hands:1.3), horror, geometry, bad_prompt, (bad hands), (((missing fingers))), multiple limbs, bad anatomy, ((morbid)), (interlocked fingers:1.2), (((Ugly Fingers))), (extra digit and hands and fingers and legs and arms:1.4), (deformed fingers:1.2), (long fingers:1.2), (bad-artist-anime), bad-artist, bad hand""",
"""(nsfw),(nude),(Nipple:2), (Cleavage:0.5),(text:2.0),(logo:2.0),(title:2.0),(worst quality:2.0), (Nevus:1.5), age spot, Acne, ((Black patches appear on the face)), skin spots, (disfigured:1.2), (low quality:2.0), (low quality:2), (deformed), ((text)), (((title))), ((logo)), (deformed, distorted, disfigured:1.3), (duplicate:2.0), distorted, (((tranny))), (unclear eyes:1.331), poorly drawn, bad anatomy, wrong anatomy, full Body, extra limb, skin blemishes, missing limb, floating limbs, (mutated hands and fingers:1.4), Face occlusion, ((Multiple people)), disconnected limbs, extra limbs, mutation, (bad proportions:1.3), mutated, ugly, disgusting, amputation, duplicate, bugly, huge eyes, monochrome, (mutilated:1.2), ((worst face)), (bad and mutated hands:1.3), horror, geometry, bad_prompt, (bad hands), (((missing fingers))), multiple limbs, bad anatomy, ((morbid)), (interlocked fingers:1.2), (((Ugly Fingers))), (extra digit and hands and fingers and legs and arms:1.4), (deformed fingers:1.2), (long fingers:1.2), (bad-artist-anime), bad-artist, bad hand""",
]


# images = pipe(prompt=prompt
# 	, negative_prompt = negative_prompt
# 	, image=init_image
# 	, num_inference_steps=30
# 	, guidance_scale=7
# 	, strength=0.65).images
a = time.time()

generator = torch.manual_seed(2)
output = pipe(
    prompt*10,
    init_image,
    negative_prompt=negative_prompt*10,
    num_inference_steps=20,
    generator=generator,
)
print("出图耗时: {}".format(time.time()-a))


# for i in range(0, len(images)):
# 	images[i].save("img2img_fantasy_landscape_{}.png".format(i))
