python img2img_universe_hr.py \
        --model_path /dnwx/datasets/models/Stable-diffusion/majicmixRealistic_v6 \
        --vae_path /dnwx/datasets/models/Stable-diffusion/vae \
        --W 848 \
        --H 1280 \
        --steps 30 \
        --scale 7 \
        --strength 0.3 \
        --outdir /root/limiao/result/boy \
        --from_file ./prompts_face.txt \
        --num_images_per_prompt 1 \
        --image_path /root/limiao/result/boy/333.png \
        --AD_Fix True \
        --face_model "./face_yolov8n_v2.pt" \
        --lora_path "/dnwx/datasets/models/Lora/" \
