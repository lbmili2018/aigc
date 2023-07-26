python img2img_sd_control_lora_hr.py \
	--model_path /dnwx/datasets/models/Stable-diffusion/majicmixRealistic_v6 \
	--vae_path /dnwx/datasets/models/Stable-diffusion/vae \
	--W 720 \
	--H 1280 \
	--steps 30 \
	--scale 7 \
	--image_path /root/limiao/input/20_universe/girl/01.png \
	--outdir /root/limiao/result/boy \
	--from_file  ./prompts_lora.txt \
	--repeats 1 \
	--num_images_per_prompt 1 \
  --controlnet canny \
  --control_scale 1.0 \
  --lora_path /dnwx/datasets/models/Lora/XSarchitectural-40PopmartMechaGirl.safetensors \
  --strength 0.5