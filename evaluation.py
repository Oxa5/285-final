# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 17:59:20 2026

@author: Oxas
"""
import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

clip_model_id = "openai/clip-vit-base-patch32"
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = CLIPProcessor.from_pretrained(clip_model_id)
model = CLIPModel.from_pretrained(clip_model_id).to(device)

def calculate_clip_score(image_path, prompt):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        return None

    inputs = processor(text=[prompt], images=image, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        score = outputs.logits_per_image.item()
        
    return score

eval_prompt = "A highly detailed professional clinical radiograph of a human thorax, frontal view, showing clear rib cage, clavicles, lungs, spine and heart silhouette, grayscale medical X-ray imaging"

before_folder = r"E:\真 学习\files\ucsd\285\final\before"
after_folder = r"E:\真 学习\files\ucsd\285\final\after"

total_score_before = 0.0
valid_count_before = 0

for i in range(1, 61):
    image_path = os.path.join(before_folder, f"before_result_{i}.png")
    score = calculate_clip_score(image_path, eval_prompt)
    
    if score is not None:
        total_score_before += score
        valid_count_before += 1

avg_score_before = total_score_before / valid_count_before if valid_count_before > 0 else 0.0

total_score_after = 0.0
valid_count_after = 0

for i in range(1, 61):
    image_path = os.path.join(after_folder, f"after_result_{i}.png")
    score = calculate_clip_score(image_path, eval_prompt)
    
    if score is not None:
        total_score_after += score
        valid_count_after += 1

avg_score_after = total_score_after / valid_count_after if valid_count_after > 0 else 0.0