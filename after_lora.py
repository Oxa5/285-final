# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 14:45:50 2026

@author: Oxas
"""
import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel
from transformers import CLIPProcessor, CLIPModel

local_model_path = r"D:\HuggingFace_Cache\models--runwayml--stable-diffusion-v1-5\snapshots\451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
lora_weights_path = r"E:\真 学习\files\ucsd\285\60_lora"
lora_scale = 0.3

lora_pipeline = StableDiffusionPipeline.from_pretrained(
    local_model_path,
    torch_dtype=torch.float16,
    local_files_only=True
).to("cuda")
lora_pipeline.safety_checker = None

lora_pipeline.unet = PeftModel.from_pretrained(lora_pipeline.unet, lora_weights_path)
lora_pipeline.unet.add_weighted_adapter(["default"], [lora_scale], "scaled_lora")
lora_pipeline.unet.set_adapter("scaled_lora")
lora_pipeline.to("cuda")

clip_model_id = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(clip_model_id).to("cuda")
clip_processor = CLIPProcessor.from_pretrained(clip_model_id)

test_prompt = "A highly detailed professional clinical radiograph of a human thorax, frontal view, showing clear rib cage, clavicles, lungs, spine and heart silhouette, grayscale medical X-ray imaging, in sks style"
neg_prompt = "wooden box, treasure chest, furniture, container, colorful, text, mutated bones, distorted skeleton, 3d render, illustration"
eval_prompt = "A highly detailed professional clinical radiograph of a human thorax, frontal view, showing clear rib cage, clavicles, lungs, spine and heart silhouette, grayscale medical X-ray imaging"
num_images = 60
total_score = 0.0

for i in range(num_images):
    current_seed = 42 + i
    generator = torch.Generator(device="cuda").manual_seed(current_seed)
    
    lora_image = lora_pipeline(
        test_prompt, 
        negative_prompt=neg_prompt,
        num_inference_steps=30, 
        guidance_scale=7.5,
        generator=generator
    ).images[0]
    
    save_path = rf"E:\真 学习\files\ucsd\285\final\after\after_result_{i+1}.png"
    lora_image.save(save_path)
    
    inputs = clip_processor(text=[eval_prompt], images=lora_image, return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        outputs = clip_model(**inputs)
        score = outputs.logits_per_image.item()
        
    total_score += score

average_score = total_score / num_images