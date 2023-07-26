python img2img_universe.py \
    --model_path /dnwx/datasets/models/Stable-diffusion/diff \
    --vae_path /dnwx/datasets/models/Stable-diffusion/vae \
    --steps 20 --scale 9 \
    --W 540 --H 720 \
    --image_path /root/limiao/boy/20230606-154758.jpg \
    --strength 0.5 \
    --outdir /root/limiao/result/res_universe \
    --from_file  ./prompts_test.txt \

#    --W 1079 --H 1438 \
