import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
import torch.nn.functional as F

# 加载预训练的自动编码器模型
vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")
# 加载预训练的CLIP文本编码器模型
text_encoder = CLIPTextModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="text_encoder")
tokenizer = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer")
# 初始化U-Net模型
unet = UNet2DConditionModel(**model_config)  # model_config为模型参数配置
# 定义扩散过程的scheduler
noise_scheduler = DDPMScheduler(
    beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000
)

# 冻结自动编码器和文本编码器参数
vae.requires_grad_(False)
text_encoder.requires_grad_(False)

opt = torch.optim.AdamW(unet.parameters(), lr=1e-4)  # 优化器

for step, batch in enumerate(train_dataloader):
    with torch.no_grad():
        # 将图像编码到潜在空间
        latents = vae.encode(batch["image"]).latent_dist.sample()
        latents = latents * vae.config.scaling_factor  # 重新缩放潜在向量
        # 获取文本嵌入
        text_input_ids = tokenizer(
            batch["text"],
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).input_ids
        text_embeddings = text_encoder(text_input_ids)[0]

    # 采样噪声
    noise = torch.randn_like(latents)
    bsz = latents.shape[0]
    # 随机采样时间步长
    timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
    timesteps = timesteps.long()

    # 向潜在向量添加噪声,进行扩散
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    # 预测噪声并计算损失
    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states=text_embeddings).sample
    loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

    loss.backward()  # 反向传播
    opt.step()  # 更新参数
    opt.zero_grad()  # 梯度清零

# 计算scaled linear noise scheduler的beta参数
betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2