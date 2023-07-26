python img2img_inpaint.py \
        --model_path  /dnwx/datasets/models/Stable-diffusion/majicmixRealistic_v6 \
        --vae_path /dnwx/datasets/models/Stable-diffusion/vae \
        --W 936 \
        --H 1664 \
        --steps 30 \
        --scale 7 \
        --image_path /root/limiao/result/boy/333.png \
        --strength 0.4 \
        --outdir /root/limiao/result \
        --from_file ./prompts_face.txt \
        --num_images_per_prompt 1 \
#        --lora_path "/dnwx/datasets/models/Lora/" \


# --model_path /dnwx/datasets/models/Stable-diffusion/majicmixRealistic_v6 \
# runwayml/stable-diffusion-inpainting
# /root/.cache/huggingface/hub/models--runwayml--stable-diffusion-inpainting