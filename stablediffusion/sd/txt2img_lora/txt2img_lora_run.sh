python txt2img_lora_hr.py \
	--model_path /dnwx/datasets/models/Stable-diffusion/majicmixRealistic_v6 \
	--W 512 \
	--H 768 \
	--steps 30 \
	--scale 7 \
	--outdir /root/limiao/result/boy \
	--from_file  ../prompts_2lora.txt \
	--num_images_per_prompt 1 \
	--sampler "euler_a" \
	--lora_path "/dnwx/datasets/models/Lora/" \
	--Highres_strength 0.5 \
	--Highres_scale 2 \
#	--vae_path /dnwx/datasets/models/Stable-diffusion/vae \
