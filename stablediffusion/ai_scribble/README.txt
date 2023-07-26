
        --sd_path   "/dnwx/datasets/models/checkpoints/checkpoints/3Guofeng3_v33.safetensors" \      

        --ctl_path  "/dnwx/datasets/lllyasviel/sd-controlnet-scribble" \ 

        --vae_path "/dnwx/datasets/models/Stable-diffusion/vae" \  				# vae-ft-mse-840000-ema-pruned

        --image_path "/root/limiao/ai_scribble/input/little girl_2225829753.jpg" \			# 输入图片：一张简笔画

        --outdir "/root/limiao/ai_scribble/input/girl_2225829753" \				# 输出路径：返回5张 512*640图片，并选取非黑值图，resize到原图作为返回值。

        --from_file "/root/limiao/ai_scribble/prompt.txt" \					# （可选）prompt默认空， 可以是英文单词，也可以是英文句子

        --seed 2023									# （可选）seed 默认空，即随机种子


pip install diffusers==0.17.0

conda activate ai_scribble
