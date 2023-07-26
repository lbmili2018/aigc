python gen_img_diffusers.py --ckpt /root/algo/stable-diffusion-webui/models/Stable-diffusion/GuoFeng3/GuoFeng3.2.safetensors \
    --outdir /root/limiao/result/res_video/guofeng --xformers --fp16 --scale 7 --sampler euler_a --steps 20 \
    --image_path /root/limiao/DeforumStableDiffusionLocal/output/2023-05/WAVE_ext1_stren06_euler_a_512_512/inputframes/ --strength 0.4 \
    --H 768 --W 512 \
    --seed  12312321313 \
    --prompt "Spirit, flowers in clusters, Transparent wings,[Pink|Light yellow|Pink-orange|Light green|Light blue|Lavender] Flowers in the background,[[Background virtual]],Forest,Hand bouquet, hair bow, hair ornament, hair ornament, hair bow, jewelry, flower on head, head wreath, (pointy ears:1.3), crystal earrings, crystal earrings, light smile, happy, --version niji, HD --style, 8k --hd, god rays, vignetting, bloom, Surrealism, rococo style, Romanticism, Baroque, Luminism, Surrealism, UHD, anatomically correct, textured skin, super detail, high details, best quality, high quality, 1080P, retina, 16k,Upright view --n (nsfw),(nude),(Nipple:2), (Cleavage:0.5),(text:2.0),(logo:2.0),(title:2.0),(worst quality:2.0), (Nevus:1.5), age spot, Acne, ((Black patches appear on the face)), skin spots, (disfigured:1.2), (low quality:2.0), (low quality:2), (deformed), ((text)), (((title))), ((logo)), (deformed, distorted, disfigured:1.3), (duplicate:2.0), distorted, (((tranny))), (unclear eyes:1.331), poorly drawn, bad anatomy, wrong anatomy, full Body, extra limb, skin blemishes, missing limb, floating limbs, (mutated hands and fingers:1.4), Face occlusion, ((Multiple people)), disconnected limbs, extra limbs, mutation, (bad proportions:1.3), mutated, ugly, disgusting, amputation, duplicate, bugly, huge eyes, monochrome, (mutilated:1.2), ((worst face)), (bad and mutated hands:1.3), horror, geometry, bad_prompt, (bad hands), (((missing fingers))), multiple limbs, bad anatomy, ((morbid)), (interlocked fingers:1.2), (((Ugly Fingers))), (extra digit and hands and fingers and legs and arms:1.4), (deformed fingers:1.2), (long fingers:1.2), (bad-artist-anime), bad-artist, bad hand"  \
    --batch_size 1 --images_per_prompt 271 --use_original_file_name \
    --control_net_models /root/algo/stable-diffusion-webui/models/ControlNet/ControlNet/models/control_sd15_hed.pth \
    --control_net_weights 1.0 --guide_image_path /root/limiao/DeforumStableDiffusionLocal/output/2023-05/WAVE_ext1_stren06_euler_a_512_512/save_hed/ --control_net_ratios 1.0 \


#    --vgg16_guidance_scale 250 --guide_image_path /root/limiao/DeforumStableDiffusionLocal/output/2023-05/WAVE_ext1_stren06_euler_a_512_512/inputframes/
#    --vae /root/algo/stable-diffusion-webui/models/VAE/anything-v4.0.vae/anything-v4.0.vae.pt \
#    --control_net_models /root/algo/stable-diffusion-webui/models/ControlNet/ControlNet/models/control_sd15_hed.pth /root/algo/stable-diffusion-webui/models/ControlNet/ControlNet/models/control_sd15_depth.pth \
#    --control_net_weights 1.0 1.0 --guide_image_path /root/limiao/test  --control_net_ratios 1.0 1.0 \


