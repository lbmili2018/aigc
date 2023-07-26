#!/bin/bash

ROOT="/root/limiao/controlnetvideo_CFAttn/user"
INPUT_NAME="black"
TEMP="${ROOT}/temp"
mkdir $TEMP
chmod 777 $TEMP

# phase 1
Videos_path1="${ROOT}/${INPUT_NAME}.mp4"
Frames_save_path1="${TEMP}/${INPUT_NAME}_half"
Video_save_path1="${TEMP}/${INPUT_NAME}.mp4"
echo $Frames_save_path1 $Video_save_path1
EXTRACT1=1

# phase 2 SD
SD_MODEL='/dnwx/datasets/models/Stable-diffusion/ghostmix_v12'
SD_VAE='/dnwx/datasets/models/Stable-diffusion/vae'
OUT_DIR="${TEMP}/${INPUT_NAME}_res11_075_all"
mkdir $OUT_DIR

# phase 3 RIFE
NEW_OUT_DIR="${ROOT}/${INPUT_NAME}_half_ghost_035_RIFE.mp4"


timer_start=`date "+%Y-%m-%d %H:%M:%S"`
echo "开始时间：$timer_start"

#python3 preprocess.py \
#        --EXTRACT_RIFE \
#        --videos_path $Videos_path1 \
#        --frames_save_path $Frames_save_path1 \
#        --video_save_path  $Video_save_path1 \
#        --time_interval $EXTRACT1

python3 img2video_CrossFrameAttn.py \
        $Video_save_path1 \
        --sd-model $SD_MODEL \
        --sd-vae $SD_VAE  \
        --controlnet hed \
        --prompt '1girl , masterpiece, top quality, best quality' \
        --negative-prompt 'nsfw, nude,masculinity,worst quality, acne, worst face, disfigured, duplicate, wrong anatomy, bad proportions,extra limb, skin blemishes, missing limb, deformed fingers, mutated hands and fingers, disconnected limbs,ugly,mutilated, bad and mutated hands, multiple limbs' \
        --prompt-strength 7 \
        --init-image-strength 0.5 \
        --controlnet-strength 0.89 \
        --color-amount 0.2 \
        --feedthrough-strength 0.5 \
        --num-inference-steps 20 \
        --start-time 0.0 \
        $OUT_DIR

#cd ECCV2022-RIFE
#python inference_video.py \
#        --exp=1 \
#        --video=$OUT_DIR \
#        --output=$NEW_OUT_DIR \
#        --fps=24
#cd ..
#rm -r $TEMP

timer_end=`date "+%Y-%m-%d %H:%M:%S"`
echo "结束时间：$timer_end"
start_seconds=$(date --date="$timer_start" +%s);
end_seconds=$(date --date="$timer_end" +%s);
echo "本次运行时间："$((end_seconds-start_seconds))"s"

