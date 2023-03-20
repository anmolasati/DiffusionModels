import torch
import torch.nn as nn 
from models  import Diffusion
from models  import UNet 
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import yaml 
import random 
from torch.utils.data import  DataLoader
from utils import BitmojiDataset
from torchmetrics.image.fid import FrechetInceptionDistance

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = UNet (device=device)
model.load_state_dict(torch.load("./results/model.pth"))

data = BitmojiDataset(root_dir=f"data/bitmojis/val", image_size=64)
dataloader = DataLoader(data, batch_size=1, shuffle=True)

diffusion = Diffusion(device=device)

num_val_imgs = 100 

real_images = []
for i in range(num_val_imgs):
    img = next(iter(dataloader)).to(device)
    real_images.append(img)

real_images = torch.cat(real_images,dim=0)
real_images = real_images.reshape(num_val_imgs,3,64,64)
real_images = (real_images.clamp(-1, 1) + 1) / 2
real_images = (real_images * 255).type(torch.uint8)

generated_imges = diffusion.sample(model,n=100)


fid = FrechetInceptionDistance(feature=2048)
fid.update(real_images, real=True)
fid.update(generated_imges, real=False)
fid_score = fid.compute()

print(f"fid score  {fid_score}")
