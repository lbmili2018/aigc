import torch

import os
from safetensors.torch import load_file
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers.models.attention_processor import AttnProcessor2_0
import copy


torch.set_grad_enabled(False)
MODEL_ID = os.environ.get("MODEL_ID")
PIPELINE = os.environ.get("PIPELINE")
VAE_ID = os.environ.get("VEA_ID")
SAFE_CHECKER = os.environ.get("SAFE_CHECKER")

class Model:
    def __init__(self):
        # self.model_id = MODEL_ID
        self.device = "cuda"
        # self.pipe = self.init_model(self.model_id, "img2img")
        # self.pipe.safety_checker = lambda images, clip_input: (images, False)
        # if VAE_ID is not None:
        #     self.set_vae(VAE_ID)
        # if SAFE_CHECKER == "true":
        #     self.set_safe_checker(SAFE_CHECKER)


    def init_model(self, MODEL_ID, PIPE_TYPE):
        """
        初始化model
        """
        if not os.path.exists(MODEL_ID):
            print("INIT MODEL ERROR: MODEL_ID is not exists, MODEL_PATH: {}".format(MODEL_ID))
            return None
        else:
            print("INIT MODEL PATH SUCCESSFUL!!!")
        if PIPE_TYPE == "img2img":
            from diffusers import StableDiffusionImg2ImgPipeline
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(MODEL_ID, revision="fp16", torch_dtype=torch.float16).to(self.device)
        pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_model_cpu_offload()
        return pipe
    

    def set_vae(self, VAE_ID):
        """
        设置VAE
        """
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(VAE_ID, subfolder="vae", torch_dtype=torch.float16).to(self.device)
        self.pipe.vae = vae


    def set_scheduler(self, scheduler):
        """
        设置scheduler, 默认euler_a
        """
        if scheduler == "euler_a":
            from diffusers import EulerAncestralDiscreteScheduler
            self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_pretrained(self.model_id, subfolder="scheduler")
        elif scheduler == "euler":
            from diffusers import EulerDiscreteScheduler
            self.pipe.scheduler = EulerDiscreteScheduler.from_pretrained(self.model_id, subfolder="scheduler")
        # elif scheduler == "DPM++":
        #     from diffusers import DPMSolverMultistepScheduler
        #     self.pipe.scheduler = DPMSolverMultistepScheduler.from_pretrained(self.model_id, subfolder="scheduler")
        else:
            from diffusers import EulerAncestralDiscreteScheduler
            self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_pretrained(self.model_id, subfolder="scheduler")
    
    
    def set_safe_checker(self, if_work):
        """
        是否开启出图安全检查
        """
        if if_work:
            self.pipe.safety_checker = lambda images, clip_input: (images, True)
        else:
            self.pipe.safety_checker = lambda images, clip_input: (images, False)
        

    def set_lora(self, pipe, lora_path, weight=1):
        # load lora weight
        state_dict = load_file(lora_path)

        LORA_PREFIX_UNET = 'lora_unet'
        LORA_PREFIX_TEXT_ENCODER = 'lora_te'

        alpha = 0.75 * weight

        visited = []

        # directly update weight in diffusers model
        for key in state_dict:
            
            # it is suggested to print out the key, it usually will be something like below
            # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"
            
            # as we have set the alpha beforehand, so just skip
            if '.alpha' in key or key in visited:
                continue
                
            if 'text' in key:
                layer_infos = key.split('.')[0].split(LORA_PREFIX_TEXT_ENCODER+'_')[-1].split('_')
                curr_layer = pipe.text_encoder
            else:
                layer_infos = key.split('.')[0].split(LORA_PREFIX_UNET+'_')[-1].split('_')
                curr_layer = pipe.unet

            # find the target layer
            temp_name = layer_infos.pop(0)
            while len(layer_infos) > -1:
                try:
                    curr_layer = curr_layer.__getattr__(temp_name)
                    if len(layer_infos) > 0:
                        temp_name = layer_infos.pop(0)
                    elif len(layer_infos) == 0:
                        break
                except Exception:
                    if len(temp_name) > 0:
                        temp_name += '_'+layer_infos.pop(0)
                    else:
                        temp_name = layer_infos.pop(0)
            
            # org_forward(x) + lora_up(lora_down(x)) * multiplier
            pair_keys = []
            if 'lora_down' in key:
                pair_keys.append(key.replace('lora_down', 'lora_up'))
                pair_keys.append(key)
            else:
                pair_keys.append(key)
                pair_keys.append(key.replace('lora_up', 'lora_down'))
            
            # update weight
            if len(state_dict[pair_keys[0]].shape) == 4:
                weight_up = state_dict[pair_keys[0]].squeeze(3).squeeze(2).to(torch.float32)
                weight_down = state_dict[pair_keys[1]].squeeze(3).squeeze(2).to(torch.float32)
                curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
            else:
                weight_up = state_dict[pair_keys[0]].to(torch.float32)
                weight_down = state_dict[pair_keys[1]].to(torch.float32)
                curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down)
                
            # update visited list
            for item in pair_keys:
                visited.append(item)

    def clear_lora(self, lora_path, weight=1):
        # load lora weight
        state_dict = load_file(lora_path)

        LORA_PREFIX_UNET = 'lora_unet'
        LORA_PREFIX_TEXT_ENCODER = 'lora_te'

        alpha = 0.75 * weight

        visited = []

        # directly update weight in diffusers model
        for key in state_dict:
            
            # it is suggested to print out the key, it usually will be something like below
            # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"
            
            # as we have set the alpha beforehand, so just skip
            if '.alpha' in key or key in visited:
                continue
                
            if 'text' in key:
                layer_infos = key.split('.')[0].split(LORA_PREFIX_TEXT_ENCODER+'_')[-1].split('_')
                curr_layer = self.pipe.text_encoder
            else:
                layer_infos = key.split('.')[0].split(LORA_PREFIX_UNET+'_')[-1].split('_')
                curr_layer = self.pipe.unet

            # find the target layer
            temp_name = layer_infos.pop(0)
            while len(layer_infos) > -1:
                try:
                    curr_layer = curr_layer.__getattr__(temp_name)
                    if len(layer_infos) > 0:
                        temp_name = layer_infos.pop(0)
                    elif len(layer_infos) == 0:
                        break
                except Exception:
                    if len(temp_name) > 0:
                        temp_name += '_'+layer_infos.pop(0)
                    else:
                        temp_name = layer_infos.pop(0)
            
            # org_forward(x) + lora_up(lora_down(x)) * multiplier
            pair_keys = []
            if 'lora_down' in key:
                pair_keys.append(key.replace('lora_down', 'lora_up'))
                pair_keys.append(key)
            else:
                pair_keys.append(key)
                pair_keys.append(key.replace('lora_up', 'lora_down'))
            
            # update weight
            if len(state_dict[pair_keys[0]].shape) == 4:
                weight_up = state_dict[pair_keys[0]].squeeze(3).squeeze(2).to(torch.float32)
                weight_down = state_dict[pair_keys[1]].squeeze(3).squeeze(2).to(torch.float32)
                curr_layer.weight.data -= alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
            else:
                weight_up = state_dict[pair_keys[0]].to(torch.float32)
                weight_down = state_dict[pair_keys[1]].to(torch.float32)
                curr_layer.weight.data -= alpha * torch.mm(weight_up, weight_down)
                
            # update visited list
            for item in pair_keys:
                visited.append(item)