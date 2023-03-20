from torch.nn import nn 
import torch 
import math 

class PositionalEncoding(nn.Module):
    def __init__(self,emb_dim,device):
        super().__init__()
        self.emb_dim = emb_dim 
        self.device = device

    def forward(self,timesteps):
        half_dim = self.emb_dim // 2
        emb = math.log(10000) / ( half_dim -1 )
        emb = torch.exp(torch.arange(half_dim,device=self.device)* -emb)
        emb = timesteps.float()[:,None] *emb[None,:]
        emb = torch.cat((torch.sin(emb),torch.cos(emb)),dim=-1)

        return emb

class ConvLayers(nn.Module):
    def __init__(self,in_channel,out_channel,mid_channel=None):
        super().__init__()
        if not mid_channel:
            mid_channel = out_channel 
        self.conv = nn.Sequential(
                                nn.Conv2d(in_channel,mid_channel,kernel_size=3, padding=1, bias=False),
                                nn.GroupNorm(1,mid_channel),
                                nn.Conv2d(mid_channel,out_channel,kernel_size=3, padding=1, bias=False)
                                nn.GroupNorm(1,out_channel),
                                nn.SiLU()
                                ) 
    def forward(self,x):
        return self.conv(x) 
    
class Up(nn.Module):
    def __init__(self,in_channel,out_channel,time_dim):
        super().__init__() 

        self.up = nn.Upsample(scale_factor=2,mode="bilinear",align_corners=True)
        self.conv = ConvLayers(in_channel,out_channel) 
        self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_dim,out_channel)
                )
    def forward(self,x,skip_x,t):
        x = self.up(x)
        x = torch.cat([skip_x,x],dim=1)
        x = self.conv(x) 
        t_emb = self.time_mlp(t)
        t_emb = t_emb.repeat(1,1,x.shape[-2],x.shape[-1])

        return x + t_emb
    
class Down(nn.Module):
    def __init__(self, in_channel,out_channel,time_dim):
        super().__init__()
        self.max_pool = nn.MaxPool2d(2)
        self.conv = ConvLayers(in_channel,out_channel) 
        self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_dim,out_channel)
                )
        
    def forward(self,x,t):
        x = self.max_pool(x) 
        x = self.conv(x) 
        t_emb = self.time_mlp(t)
        t_emb = t_emb.repeat(1,1,x.shape[-2],x.shape[-1])

        return x  + t_emb 
    
    
class UNet(nn.Module):
    def __init__(self,in_channel=3,out_channel=3,base_channel=64,time_dim=256,device=None):
        super().__init__()
        self.init_conv  = ConvLayers(in_channel,base_channel,time_dim) 
        self.down1 =  Down(base_channel,base_channel*2,time_dim)  
        self.down2 = Down(base_channel*2,base_channel*4,time_dim)
        self.down3 = Down(base_channel*4,base_channel*4,time_dim)

        self.conv1 = ConvLayers(base_channel*4,base_channel*8)
        self.conv2 = ConvLayers(base_channel*8,base_channel*8)
        self.conv3 = ConvLayers(base_channel*8,base_channel)

        self.up1 = Up(base_channel*8,base_channel*2,time_dim) 
        self.up1 = Up(base_channel*4,base_channel,time_dim) 
        self.up3 = Up(base_channel*2,base_channel,time_dim) 
    
        self.final_conv = nn.Con2d(base_channel,out_channel,kernel_size=1) 


        self.time_dim = time_dim 

        self.sinus_pos_encoding = PositionalEncoding(time_dim,device) 


    def forward(self,x,t): 
        t = t.unsqueeze(-1).type(torch.float) 
        t = self.sinus_pos_encoding(t) 
        x1 = self.init_conv(x) 
        x2 = self.down1(x1,t)
        x3 = self.down2(x2,t)
        x4 = self.down3(x3,t)

        x5 = self.conv1(x4)
        x5 = self.conv2(x5)
        x5 = self.conv3(x5)

        x6 = self.up1(x5,x3,t)
        x6 = self.up2(x6,x2,t)
        x6 = self.up3(x6,x1,t)

        out = self.final_conv(x6) 

        return out 








