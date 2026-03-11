# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 12:31:28 2026

@author: Oxas
"""
import os
cache_dir = r"D:\HuggingFace_Cache"

import torch
from diffusers import StableDiffusionPipeline

model_id = "runwayml/stable-diffusion-v1-5"

pipeline = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16
)