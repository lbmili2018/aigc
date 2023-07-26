python txt2img_multicontrol_lora_hr.py \
	--model_path /dnwx/datasets/models/Stable-diffusion/majicmixRealistic_v6 \
	--vae_path /dnwx/datasets/models/Stable-diffusion/vae \
	--W 720 \
	--H 1280 \
	--steps 30 \
	--scale 7 \
	--outdir /root/limiao/result/boy \
	--from_file  ../prompts_lora.txt \
	--repeats 1 \
	--num_images_per_prompt 1 \
	--sampler "euler_a" \
	--controlnet "hed" \
  --control_scale "0.8" \
	--lora_path "/dnwx/datasets/models/Lora/" \

