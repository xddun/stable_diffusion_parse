import torch
from diffusers import AutoencoderKL
import numpy as np
from PIL import Image

# 加载模型: autoencoder可以通过SD权重指定subfolder来单独加载
autoencoder = AutoencoderKL.from_pretrained(
    "/ssd/xiedong/src_data/eff_train/Stable-diffusion/majicmixRealistic_v7_diffusers", subfolder="vae")
autoencoder.to("cuda", dtype=torch.float16)

# 读取图像并预处理
raw_image = Image.open("girl.png").convert("RGB").resize((512, 512))
image = np.array(raw_image).astype(np.float32) / 127.5 - 1.0
image = image[None].transpose(0, 3, 1, 2)
image = torch.from_numpy(image)

# 压缩图像为latent并重建
with torch.inference_mode():
    # latentx形状是 (B, C, H, W) 的张量  (1,4,64,64)
    latentx = autoencoder.encode(image.to("cuda", dtype=torch.float16)).latent_dist.sample()  # 压缩
    # 保存 latent 为 PNG 图像
    latent = latentx.permute(0, 2, 3, 1)  # 将 latent 重新排列为 (B, H, W, C) 格式
    latent = latent.cpu().numpy()  # 将 latent 转换为 NumPy 数组
    latent = (latent * 127.5 + 127.5).astype('uint8')  # 将值缩放到 [0, 255] 范围内,并转换为 uint8 类型
    latent = latent.squeeze(0)  # 去掉批次维度

    latent_image = Image.fromarray(latent)  # 将 NumPy 数组转换为 PIL Image
    latent_image.save("latent.png")  # 保存为 PNG 图像
    # shape
    print(latentx.shape)
    rec_image = autoencoder.decode(latentx).sample
    rec_image = (rec_image / 2 + 0.5).clamp(0, 1)
    rec_image = rec_image.cpu().permute(0, 2, 3, 1).numpy()
    rec_image = (rec_image * 255).round().astype("uint8")
    rec_image = Image.fromarray(rec_image[0])

# save
rec_image.save("demo.png")
