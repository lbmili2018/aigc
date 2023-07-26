#python gen_img_diffusers.py --ckpt /root/algo/stable-diffusion-webui/models/Stable-diffusion/WAVE-P.ckpt \
#    --outdir /root/limiao/result/res_video_manhua_test/ --xformers --fp16 --scale 7 --sampler euler_a --steps 20 \
#    --image_path /root/limiao/DeforumStableDiffusionLocal/output/2023-05/WAVE_ext1_stren06_euler_a_512_512/inputframes/ --strength 0.4 \
#    --H 768 --W 512 \
#    --seed  12312321313 \
#    --prompt "best quality, masterpiece, highres, china dress,Beautiful face,upon_body, tyndall effect,photorealistic,atmospheric perspective, dark studio, rim lighting, two tone lighting,(high detailed skin:1.2), 8k uhd, dslr, soft lighting, high quality, volumetric lighting, candid, Photograph, high resolution, 4k, 8k, Bokeh   --n ((simple background)),monochrome ,lowres, bad anatomy, (bad hands), text, error, (missing fingers), extra digits, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, ugly,pregnant,vore,duplicate,morbid,mutilated,transexual, hermaphrodite,long neck,mutated hands,poorly drawn hands,poorly drawn face,mutation,deformed,bad proportions,malformed limbs,extra limbs,cloned face,disfigured,gross proportions, (missing arms),(missing legs), (extra arms),(extra legs),pubic hair, plump,bad legs,error legs,bad feet" \
#    --batch_size 1 --images_per_prompt 271 --use_original_file_name \
#    --control_net_models /root/algo/stable-diffusion-webui/models/ControlNet/ControlNet/models/control_sd15_hed.pth \
#    --control_net_weights 1.0 --guide_image_path /root/limiao/DeforumStableDiffusionLocal/output/2023-05/WAVE_ext1_stren06_euler_a_512_512/save_hed/ --control_net_ratios 1.0 \


python gen_img_diffusers.py --ckpt /root/algo/stable-diffusion-webui/models/Stable-diffusion/CounterfeitV25_25.safetensors  \
    --outdir /root/limiao/result/res_video_Counter055_mfr/ --xformers --fp16 --scale 9 --sampler euler_a --steps 30 \
    --image_path /root/limiao/DeforumStableDiffusionLocal/output/2023-05/WAVE_ext1_stren06_euler_a_512_512/inputframes/ --strength 0.55 \
    --H 768 --W 512 \
    --seed 386461727 \
    --prompt "paint splashes, splatter, outrun, vaporware, shaded flat illustration, digital art, trending on artstation, highly detailed, fine detail, intricate, <lora:Colorwater_v4:0.8>  --n ((disfigured)), ((bad art)), ((deformed)), ((poorly drawn)), ((extra limbs)), ((close up)), ((b&w)), weird colors, blurry,nsfw, nude, naked, lvavaginal, nudity, topless, vulva,lowres, badanatomy, bad hands,paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glans" \
    --batch_size 1 --images_per_prompt 271 --use_original_file_name \
    --control_net_models /root/algo/stable-diffusion-webui/models/ControlNet/ControlNet/models/control_sd15_hed.pth \
    --control_net_weights 1.0 \
    --guide_image_path /root/limiao/DeforumStableDiffusionLocal/output/2023-05/WAVE_ext1_stren06_euler_a_512_512/save_hed/ --control_net_ratios 1.0 \


