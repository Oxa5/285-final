# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 19:13:16 2026

@author: Oxas
"""
import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from torchmetrics.image.fid import FrechetInceptionDistance

real_images_dir = r"E:\真 学习\files\ucsd\285\data" 
generated_images_before = r"E:\真 学习\files\ucsd\285\final\before"
generated_images_after = r"E:\真 学习\files\ucsd\285\final\after"
num_images = 60
device = "cuda" if torch.cuda.is_available() else "cpu"

preprocess = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.Lambda(lambda x: x.convert("RGB")),
    transforms.PILToTensor()
])

def load_images_to_tensor(folder_path, prefix="", is_real=False):
    image_list = []
    count = 0
    
    if not is_real:
        for i in range(1, num_images + 1):
            filepath = os.path.join(folder_path, f"{prefix}{i}.png")
            if os.path.exists(filepath):
                img = Image.open(filepath)
                image_list.append(preprocess(img))
                count += 1
    else:
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(folder_path, filename)
                img = Image.open(filepath)
                image_list.append(preprocess(img))
                count += 1
                if count >= num_images:
                    break
                    
    return torch.stack(image_list).to(device)

fid = FrechetInceptionDistance(feature=2048).to(device)
real_tensor = load_images_to_tensor(real_images_dir, is_real=True)
base_fake_tensor = load_images_to_tensor(generated_images_before, prefix="before_result_")

fid.update(real_tensor, real=True)
fid.update(base_fake_tensor, real=False)
fid_score_before = fid.compute()

fid.reset()

lora_fake_tensor = load_images_to_tensor(generated_images_after, prefix="after_result_")

fid.update(real_tensor, real=True)
fid.update(lora_fake_tensor, real=False)
fid_score_after = fid.compute()