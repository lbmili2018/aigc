python3 controlNet_image.py \
        /root/limiao/input/ai_video/input.mp4 \
        --controlnet hed \
        --prompt '1girl' \
        --prompt-strength 7 \
        --show-input \
        --show-detector \
        --show-motion \
        --dump-frames '{instem}_1111/{n:08d}.png' \
        --init-image-strength 0.75 \
        --controlnet-strength 0.89 \
        --color-amount 0.2 \
        --feedthrough-strength 0.8 \
        --show-output \
        --num-inference-steps 20 \
        --start-time 0.0 \
        '{instem}_out1111.mp4'
