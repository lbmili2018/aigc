#python gen_img_diffusers_video.py --ckpt /root/algo/stable-diffusion-webui/models/Stable-diffusion/GuoFeng3/GuoFeng3.2.safetensors \
#    --outdir /root/limiao/result/res_video_guofeng_mfr/ --xformers --fp16 --scale 7 --sampler euler_a --steps 20 \
#    --image_path /root/limiao/DeforumStableDiffusionLocal/output/2023-05/WAVE_ext1_stren06_euler_a_512_512/inputframes/ --strength 0.4 \
#    --H 768 --W 512 \
#    --seed  12312321313 \
#    --prompt "Spirit, flowers in clusters, Transparent wings,[Pink|Light yellow|Pink-orange|Light green|Light blue|Lavender] Flowers in the background,[[Background virtual]],Forest,Hand bouquet, hair bow, hair ornament, hair ornament, hair bow, jewelry, flower on head, head wreath, (pointy ears:1.3), crystal earrings, crystal earrings, light smile, happy, --version niji, HD --style, 8k --hd, god rays, vignetting, bloom, Surrealism, rococo style, Romanticism, Baroque, Luminism, Surrealism, UHD, anatomically correct, textured skin, super detail, high details, best quality, high quality, 1080P, retina, 16k,Upright view --n (nsfw),(nude),(Nipple:2), (Cleavage:0.5),(text:2.0),(logo:2.0),(title:2.0),(worst quality:2.0), (Nevus:1.5), age spot, Acne, ((Black patches appear on the face)), skin spots, (disfigured:1.2), (low quality:2.0), (low quality:2), (deformed), ((text)), (((title))), ((logo)), (deformed, distorted, disfigured:1.3), (duplicate:2.0), distorted, (((tranny))), (unclear eyes:1.331), poorly drawn, bad anatomy, wrong anatomy, full Body, extra limb, skin blemishes, missing limb, floating limbs, (mutated hands and fingers:1.4), Face occlusion, ((Multiple people)), disconnected limbs, extra limbs, mutation, (bad proportions:1.3), mutated, ugly, disgusting, amputation, duplicate, bugly, huge eyes, monochrome, (mutilated:1.2), ((worst face)), (bad and mutated hands:1.3), horror, geometry, bad_prompt, (bad hands), (((missing fingers))), multiple limbs, bad anatomy, ((morbid)), (interlocked fingers:1.2), (((Ugly Fingers))), (extra digit and hands and fingers and legs and arms:1.4), (deformed fingers:1.2), (long fingers:1.2), (bad-artist-anime), bad-artist, bad hand"  \
#    --batch_size 1 --images_per_prompt 271 --use_original_file_name \
#    --control_net_models /root/algo/stable-diffusion-webui/models/ControlNet/ControlNet/models/control_sd15_hed.pth \
#    --control_net_weights 1.0 \
#    --guide_image_path /root/limiao/DeforumStableDiffusionLocal/output/2023-05/WAVE_ext1_stren06_euler_a_512_512/save_hed/ --control_net_ratios 1.0 \


python gen_img_diffusers_video_debug.py --ckpt /root/algo/stable-diffusion-webui/models/Stable-diffusion/WAVE-P.ckpt \
    --outdir /root/limiao/result/res_video_manhua06_mfr/ --xformers --fp16 --scale 7 --sampler euler_a --steps 20 \
    --image_path /root/limiao/DeforumStableDiffusionLocal/output/2023-05/WAVE_ext1_stren06_euler_a_512_512/inputframes/ --strength 0.4 \
    --H 512 --W 512 \
    --seed  12312321313 \
    --prompt "best quality, masterpiece, highres, china dress,Beautiful face,upon_body, tyndall effect,photorealistic,atmospheric perspective, dark studio, rim lighting, two tone lighting,(high detailed skin:1.2), 8k uhd, dslr, soft lighting, high quality, volumetric lighting, candid, Photograph, high resolution, 4k, 8k, Bokeh   --n ((simple background)),monochrome ,lowres, bad anatomy, (bad hands), text, error, (missing fingers), extra digits, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, ugly,pregnant,vore,duplicate,morbid,mutilated,transexual, hermaphrodite,long neck,mutated hands,poorly drawn hands,poorly drawn face,mutation,deformed,bad proportions,malformed limbs,extra limbs,cloned face,disfigured,gross proportions, (missing arms),(missing legs), (extra arms),(extra legs),pubic hair, plump,bad legs,error legs,bad feet" \
    --batch_size 1 --images_per_prompt 271 --use_original_file_name \
    --control_net_models /root/algo/stable-diffusion-webui/models/ControlNet/ControlNet/models/control_sd15_hed.pth \
    --control_net_weights 1.0 \
    --guide_image_path /root/limiao/DeforumStableDiffusionLocal/output/2023-05/WAVE_ext1_stren06_euler_a_512_512/save_hed/ --control_net_ratios 1.0 \
#    --diffusers_xformers


#python gen_img_diffusers_video_debug.py --ckpt /root/algo/stable-diffusion-webui/models/Stable-diffusion/CounterfeitV25_25.safetensors  \
#    --outdir /root/limiao/result/res_video_too04_mfr/ --xformers --fp16 --scale 9 --sampler euler_a --steps 30 \
#    --image_path /root/limiao/DeforumStableDiffusionLocal/output/2023-05/WAVE_ext1_stren06_euler_a_512_512/inputframes/ --strength 0.4 \
#    --H 768 --W 512 \
#    --seed 386461727 \
#    --prompt "paint splashes, splatter, outrun, vaporware, shaded flat illustration, digital art, trending on artstation, highly detailed, fine detail, intricate, <lora:Colorwater_v4:0.8>  --n ((disfigured)), ((bad art)), ((deformed)), ((poorly drawn)), ((extra limbs)), ((close up)), ((b&w)), weird colors, blurry,nsfw, nude, naked, lvavaginal, nudity, topless, vulva,lowres, badanatomy, bad hands,paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glans" \
#    --batch_size 1 --images_per_prompt 271 --use_original_file_name \
#    --control_net_models /root/algo/stable-diffusion-webui/models/ControlNet/ControlNet/models/control_sd15_hed.pth \
#    --control_net_weights 1.0 \
#    --guide_image_path /root/limiao/DeforumStableDiffusionLocal/output/2023-05/WAVE_ext1_stren06_euler_a_512_512/save_hed/ --control_net_ratios 1.0 \
#    --diffusers_xformers




#    --prompt "best quality, masterpiece, highres, china dress,Beautiful face,upon_body, tyndall effect,photorealistic,atmospheric perspective, dark studio, rim lighting, two tone lighting,(high detailed skin:1.2), 8k uhd, dslr, soft lighting, high quality, volumetric lighting, candid, Photograph, high resolution, 4k, 8k, Bokeh   --n ((simple background)),monochrome ,lowres, bad anatomy, (bad hands), text, error, (missing fingers), extra digits, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, ugly,pregnant,vore,duplicate,morbid,mutilated,transexual, hermaphrodite,long neck,mutated hands,poorly drawn hands,poorly drawn face,mutation,deformed,bad proportions,malformed limbs,extra limbs,cloned face,disfigured,gross proportions, (missing arms),(missing legs), (extra arms),(extra legs),pubic hair, plump,bad legs,error legs,bad feet" \
#    --prompt "masterpiece,best quality,beautiful detailed eyes --n nsfw, nude, naked, lvavaginal, nudity, topless, vulva,lowres, badanatomy, bad hands, text, error, missing fingers, extradigit, fewerdigits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark,username,blurry,(mutated hands and fingers:1.5 ),fused breasts, bad breasts,huge breasts, poorly drawn breasts, extra breasts, liquid breasts, heavy breasts, missing breasts,huge haunch,huge thighs,huge calf,bad hands,fused hand,missing hand,multiple legs" \
#    --vgg16_guidance_scale 250 --guide_image_path /root/limiao/DeforumStableDiffusionLocal/output/2023-05/WAVE_ext1_stren06_euler_a_512_512/inputframes/
#    --vae /root/algo/stable-diffusion-webui/models/VAE/anything-v4.0.vae/anything-v4.0.vae.pt \
#    --control_net_models /root/algo/stable-diffusion-webui/models/ControlNet/ControlNet/models/control_sd15_hed.pth /root/algo/stable-diffusion-webui/models/ControlNet/ControlNet/models/control_sd15_depth.pth \
#    --control_net_weights 1.0 1.0 --guide_image_path /root/limiao/test  --control_net_ratios 1.0 1.0 \

