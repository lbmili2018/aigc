python inference.py \
    --variant mobilenetv3 \
    --checkpoint "./checkpoint/rvm_mobilenetv3.pth" \
    --device cuda \
    --input-source "/root/limiao/wuxiaomao1.mov" \
    --output-type video \
    --output-alpha "/root/limiao/alpha.mp4" \
    --output-video-mbps 4 \
    --seq-chunk 1

#    --output-composition "composition.mp4" \
#    --output-foreground "foreground.mp4" \
