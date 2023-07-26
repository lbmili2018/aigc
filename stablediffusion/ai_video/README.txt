
输入路径eg: "/root/limiao/img2video/user/xxx.mp4"

	ROOT="/root/limiao/img2video/user"

	INPUT_NAME="xxx"

	TEMP="/root/limiao/img2video/user/temp", 存放中间结果，程序执行完毕会删除该文件夹。

	NEW_OUT_DIR="${ROOT}/${INPUT_NAME}_half_ghost_035_RIFE.mp4"，已写死，最终输出结果放在"/root/limiao/controlnetvideo/user"

SD_MODEL='/dnwx/datasets/models/Stable-diffusion/diff'

SD_VAE='/dnwx/datasets/models/Stable-diffusion/vae'

EXTRACT1=2, 间隔2帧抽帧

pip install diffusers==0.16.1

conda activate ai_video
