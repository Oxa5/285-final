# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 22:02:45 2026

@author: Oxas
"""
import os
import time
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model
import bitsandbytes as bnb

model_id = "runwayml/stable-diffusion-v1-5"
device = "cuda"
image_folder = r"E:\真 学习\files\ucsd\285\data"

vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float16).to(device)
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=torch.float16).to(device)
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=torch.float16).to(device)
noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

vae.requires_grad_(False)
text_encoder.requires_grad_(False)
unet.requires_grad_(False)

unet.enable_gradient_checkpointing()

lora_config = LoraConfig(
    r=8, 
    lora_alpha=16,
    target_modules=["to_q", "to_v", "to_k", "to_out.0"],
    lora_dropout=0.1,
)
unet = get_peft_model(unet, lora_config)

optimizer = bnb.optim.AdamW8bit(unet.parameters(), lr=1e-4, weight_decay=1e-2)

class StyleDataset(Dataset):
    def __init__(self, image_dir, trigger_word="in sks style"):
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.trigger_word = trigger_word
        self.transform = transforms.Compose([
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image_tensor = self.transform(image)
        prompt = f"A picture {self.trigger_word}" 
        return image_tensor, prompt

dataset = StyleDataset(image_dir=image_folder)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

num_epochs = 15 
total_start_time = time.time()

for epoch in range(num_epochs):
    unet.train()
    epoch_loss = 0
    for batch_images, batch_prompts in dataloader:
        optimizer.zero_grad()
        
        with torch.autocast("cuda"):
            latents = vae.encode(batch_images.to(device, dtype=torch.float16)).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
            
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()
            
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            text_inputs = tokenizer(batch_prompts, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
            encoder_hidden_states = text_encoder(text_inputs.input_ids.to(device))[0]
            
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            
            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    current_time = time.time()
    elapsed_seconds = current_time - total_start_time

pipeline = StableDiffusionPipeline.from_pretrained(
    model_id,
    unet=unet,
    text_encoder=text_encoder,
    vae=vae,
    torch_dtype=torch.float16,
).to(device)

pipeline.set_progress_bar_config(disable=False)

test_prompt = "A cute cat playing with a ball in sks style" 

generated_image = pipeline(test_prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
generated_image.save(r"E:\真 学习\files\ucsd\285\final\output.png")