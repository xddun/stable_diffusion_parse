def convert_sdxl_to_diffusers(pretrained_ckpt_path, output_diffusers_path):
    import os
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 设置 HF 镜像源（国内用户使用）
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 设置 GPU 所使用的节点

    import torch
    from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline
    # pipe = StableDiffusionXLPipeline.from_single_file(pretrained_ckpt_path, torch_dtype=torch.float16).to("cuda")
    # pipe.save_pretrained(output_diffusers_path)

    pipe = StableDiffusionPipeline.from_single_file(pretrained_ckpt_path, torch_dtype=torch.float16).to("cuda")
    pipe.save_pretrained(output_diffusers_path)


# 对一个sd15分解
if __name__ == '__main__':
    convert_sdxl_to_diffusers("/ssd/xiedong/src_data/eff_train/Stable-diffusion/majicmixRealistic_v7.safetensors",
                              "/ssd/xiedong/src_data/eff_train/Stable-diffusion/majicmixRealistic_v7_diffusers")
