diffusers == 0.13.0
support  Long-Prompt Weight inference.

img2img_universe.py   main函数
pipeline_stable_diffusion_img2img.py   改动底层源码，pip install diffusers==0.13.0
prompts_test.txt   输入的promt， 格式为：正向_--neg_反向_--seed_  , 其中"_" 代表空格, 每行代表一个风格。
universe_diffuser.sh   执行脚本及参数
