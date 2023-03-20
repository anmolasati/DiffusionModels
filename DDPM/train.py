import torch
import torch.nn as nn
import torch.optim as optim
from models  import Diffusion
from models  import UNet
from utils import BitmojiDataset
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import yaml 
import random 
from torch.utils.data import  DataLoader

f = open("./config/config.yaml", 'r',encoding="utf-8" )
config_dict = yaml.load(f.read(),Loader=yaml.SafeLoader)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

seed = 369
random.seed(seed)
torch.manual_seed(seed)
print("Random Seed: ", seed)

img_size = config_dict["img_size"]
in_channel = config_dict["in_channel"]
out_channel = config_dict["out_channel"]
base_channel = config_dict["base_channel"]
beta_start =  config_dict["beta_start"]
beta_end = config_dict["beta_end"]
noise_steps = config_dict["noise_steps"]
time_dim = config_dict["time_dim"]
epochs = config_dict["epochs"]

data = BitmojiDataset(root_dir=f"data/bitmojis/train", image_size=config_dict["image_size"])
dataloader = DataLoader(data, config_dict["batch_size"], shuffle=True)

model = UNet(in_channel=in_channel,
             out_channel=out_channel,
             base_channel=base_channel,
             time_dim=time_dim,
             device=device)

diffusion = Diffusion(noise_steps = noise_steps,
                      beta_start=beta_start,
                      beta_end=beta_end,
                      img_size=img_size,
                      device=device)

optimizer = optim.AdamW(model.parameters(), lr=1e-4)

criterion = nn.MSELoss() 

for epoch in range(epochs):
    tqdm_dataloader = tqdm(dataloader)
    for batch in tqdm_dataloader:
        images = batch.to(device) 
        t = diffusion.get_timestamps(batch.shape[0]).to(device)
        x_t,noise = diffusion.forward_diffusion(images,t)

        pred_noise = model(x_t,t)
        loss = criterion(noise,pred_noise)
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step() 

        tqdm_dataloader.set_description('Train Loss {:.6f} '.format(loss))

torch.save(model.state_dict(),f"results/model.pth")







