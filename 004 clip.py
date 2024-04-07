from transformers import CLIPTextModel, CLIPTokenizer

text_encoder = CLIPTextModel.from_pretrained(
    "/ssd/xiedong/src_data/eff_train/Stable-diffusion/majicmixRealistic_v7_diffusers", subfolder="text_encoder").to(
    "cuda")
# text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to("cuda")
tokenizer = CLIPTokenizer.from_pretrained(
    "/ssd/xiedong/src_data/eff_train/Stable-diffusion/majicmixRealistic_v7_diffusers", subfolder="tokenizer")
# tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

# 对输入的text进行tokenize，得到对应的token ids
prompt = "a photograph of an astronaut riding a horse"
text_input_ids = tokenizer(
    prompt,
    padding="max_length",
    max_length=tokenizer.model_max_length,
    truncation=True,
    return_tensors="pt"
).input_ids

print(text_input_ids.shape)  # torch.Size([1, 77]
# 将token ids送入text model得到77x768的特征
text_embeddings = text_encoder(text_input_ids.to("cuda"))[0]
print(text_embeddings.shape)  # torch.Size([1, 77, 768])
