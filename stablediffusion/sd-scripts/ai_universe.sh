#python gen_img_diffusers.py --ckpt /root/algo/stable-diffusion-webui/models/Stable-diffusion/WAVE-P.ckpt \
#    --vae /dnwx/datasets/models/VAE/vae-ft-mse-840000-ema-pruned/vae-ft-mse-840000-ema-pruned.ckpt \
#    --from_file ./prompts.txt \
#    --batch_size 1 --images_per_prompt 1 --sequential_file_name \
#    --seed 1 \
#    --xformers --fp16 \
#    --W 512 --H 512 --steps 20 --scale 7 --sampler euler_a \
#    --image_path /root/limiao/girl.jpg --strength 0.4 \
#    --outdir /root/limiao/result/res_universe \
##    --diffusers_xformers \


#    --control_net_models /root/algo/stable-diffusion-webui/models/ControlNet/ControlNet/models/control_sd15_hed.pth \
#    --control_net_weights 1.0 --guide_image_path /root/limiao/girl.jpg  --control_net_ratios 1.0

#    v1-5-pruned-emaonly.ckpt
#    /dnwx/datasets/models/VAE/vae-ft-mse-840000-ema-pruned/vae-ft-mse-840000-ema-pruned.ckpt,  /root/algo/stable-diffusion-webui/models/VAE/anything-v4.0.vae/anything-v4.0.vae.pt
#    'ddim', 'pndm', 'lms', 'euler', 'euler_a', 'heun', 'dpm_2', 'dpm_2_a', 'dpmsolver', 'dpmsolver++', 'dpmsingle', 'k_lms', 'k_euler', 'k_euler_a', 'k_dpm_2', 'k_dpm_2_a'
#    --ckpt /root/algo/stable-diffusion-webui/models/Stable-diffusion/toonyou_beta2.safetensors \ WAVE-P.ckpt
#    --control_net_models /root/algo/stable-diffusion-webui/models/ControlNet/ControlNet/models/control_sd15_hed.pth \
#    --control_net_weights 1.0 --guide_image_path /root/limiao/test/000070.jpg  --control_net_ratios 1.0 \
#    --H 960 --W 544 \
#    --control_net_models /root/algo/stable-diffusion-webui/models/ControlNet/ControlNet/models/control_sd15_hed.pth /root/algo/stable-diffusion-webui/models/ControlNet/ControlNet/models/control_sd15_depth.pth \
#    --control_net_weights 1.0 1.0 --guide_image_path /root/limiao/test  --control_net_ratios 1.0 1.0 \
#    --vgg16_guidance_scale 250 --guide_image_path /root/limiao/test


#python gen_img_diffusers.py --ckpt /root/algo/stable-diffusion-webui/models/Stable-diffusion/ghostmix_v12.safetensors \
#    --vae /dnwx/datasets/models/VAE/vae-ft-mse-840000-ema-pruned/vae-ft-mse-840000-ema-pruned.ckpt \
#    --prompt "a girl" \
#    --batch_size 1 --images_per_prompt 1 --sequential_file_name \
#    --seed 4154105873 \
#    --xformers --fp16 \
#    --W 720 --H 1280 --steps 20 --scale 7 --sampler euler_a \
#    --image_path /root/limiao/orig.jpg --strength 0.75 \
#    --outdir /root/limiao/result/res_universe
##    --diffusers_xformers


python gen_img_diffusers.py --ckpt /root/algo/stable-diffusion-webui/models/Stable-diffusion/ghostmix_v12.safetensors \
    --vae /dnwx/datasets/models/VAE/vae-ft-mse-840000-ema-pruned/vae-ft-mse-840000-ema-pruned.ckpt \
    --batch_size 1 --images_per_prompt 1 --sequential_file_name \
    --seed 111 \
    --xformers --fp16 \
    --W 512 --H 768 --steps 20 --scale 7 --sampler euler_a \
    --image_path /root/limiao/orig.jpg --strength 0.5 \
    --outdir /root/limiao/result/res_universe \
    --prompt "(Fox Ears: 1.5), ([Smooth | Flawless | Pale :1.2 |white ] skin),(Facial features details:1.2),  (face focus:1.5), textured skin, Pet the baby fox,super detail, high details, (hyper quality), (Ultra HD), (hyperdetailed:1.2), best quality, high quality,masterpiece --n nsfw, nude, Nipple" \
    --diffusers_xformers
#    --prompt "(Angel with open wings:1.5), (white blond hair:1.2) , tarot, vulcan salute, Attention, halo behind head, finely_detailed,perfect_eyes,perfect_face, perfect_fingers, Golden rose ,(crown:1.2), golden light, shine --n (nsfw:2.0), (nude:2.0), (Nipple:2.0), (Cleavage:0.8), (Big breasts:1.5),(Detailed of man:2.0), (Masculine characteristics:2.0),(Masculinity:2.0),(Male:2.0),(Cross-eyed:2.0),(Beard:1.3),(beard stubble:1.5),((text)), (((title))), ((logo)),(worst quality:2.0), (Nevus:2.0), (age spot:2.0), (Acne:2.0),(skin spots:2.0), ((Black patches appear on the face)), (disfigured:1.2), (low quality:2.0), (low quality:2), (deformed),  (deformed, distorted, disfigured:1.3), (duplicate:2.0), distorted, (((tranny))), (unclear eyes:1.331), poorly drawn, bad anatomy, wrong anatomy, full Body, extra limb, skin blemishes, missing limb, floating limbs, (mutated hands and fingers:1.4), Face occlusion, ((Multiple people)), disconnected limbs, extra limbs, mutation, (bad proportions:1.3), mutated, ugly, disgusting, amputation, duplicate, bugly, huge eyes, (mutilated:1.2), ((worst face)), (bad and mutated hands:1.3), horror, geometry, bad_prompt, (bad hands), (((missing fingers))), multiple limbs, bad anatomy, ((morbid)), (interlocked fingers:1.2), (((Ugly Fingers))), (extra digit and hands and fingers and legs and arms:1.4), (deformed fingers:1.2), (long fingers:1.4), (bad-artist-anime), bad-artist, bad hand" \