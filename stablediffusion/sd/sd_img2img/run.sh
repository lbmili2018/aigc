python img2img_universe_hr.py \
        --model_path /dnwx/datasets/models/Stable-diffusion/majicmixRealistic_v6 \
        --vae_path /dnwx/datasets/models/Stable-diffusion/vae \
        --W 853 \
        --H 1280 \
        --steps 30 \
        --scale 7 \
        --strength 0.3 \
        --outdir /root/limiao/result/boy \
        --from_file ./prompts_test.txt \
        --num_images_per_prompt 1 \
        --image_path /root/limiao/input/666.png \
        --lora_path "/dnwx/datasets/models/Lora/" \

#         --model_path /dnwx/datasets/models/Stable-diffusion/majicmixRealistic_v6 \
