python gen_img_diffusers.py --ckpt /root/algo/stable-diffusion-webui/models/Stable-diffusion/WAVE-P.ckpt --outdir outputs \
    --xformers --fp16 --W 512 --H 704 --scale 12.5 --sampler k_euler_a \
    --steps 32 --batch_size 4 --images_per_prompt 16 \
    --prompt "<lora:5_17:0.7>, Portrait of young girl, dramatic lighting, illustration by greg rutkowski, yoji shinkawa, 4k, digital art, concept art, trending on artstation,masterpiece --n (nsfw),(nude),
(low quality:2),sketch, (worst quality:2.0), (low quality:2.0),(simple background),
duplicate, ugly, huge eyes, text, logo, monochrome, worst face, (bad and mutated hands:1.3), (blurry:2.0), horror, geometry, bad_prompt, (bad hands), (missing fingers), multiple limbs, bad anatomy, (interlocked fingers:1.2), Ugly Fingers, (extra digit and hands and fingers and legs and arms:1.4), crown braid, ((2girl)), (deformed fingers:1.2), (long fingers:1.2),succubus wings,horn,succubus horn,succubus hairstyle, (bad-artist-anime), bad-artist, bad hand" \
    --network_module networks.lora \
    --network_weights /root/Training_Resources/train_jn/5-model/5_17.safetensors 
