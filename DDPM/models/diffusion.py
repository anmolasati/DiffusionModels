import torch
import torch.nn.functional as F
from tqdm import tqdm

class Diffusion:
    def __init__(self, noise_steps=500, beta_start = 1e-4, beta_end= 1e-2,img_size=64,device=None) :
        self.device = device 
        self.img_size = img_size
        self.noise_steps = noise_steps 
        self.beta_Start = beta_start 
        self.beta_end = beta_end

        self.beta = self.noise_schedule().to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha,dim=0)

    def noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps) 
    
    def get_timestamps(self,n):
        return torch.randint(low=1,high=self.noise_steps,size=(n))
    
    @torch.no_grad()
    def sample(self,model,n,skip=1):

        model.eval()
        x = torch.randn((n,3,self.img_size,self.img_size)).to(self.device)

        for t_s in reversed(range(1,self.noise_steps,skip)):
            t_batch = torch.full((n,), t_s, device= self.device, dtype=torch.long)
            pred_noise = model(x,t_batch)
            alpha = self.alpha[t_batch]
            alpha_hat = self.alpha_hat[t_batch]
            beta = self.beta[t_batch]

            if t_s > 1:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            x = 1 / torch.sqrt(alpha) *(x - ((1-alpha) / (torch.sqrt(1- alpha_hat))) * pred_noise) + torch.sqrt(beta) * noise 
            
        x = (x.clamp(-1,1) +1) /2
        x = (x*255).type(torch.uint8)
        
        return x 
    
    def forward_diffusion(self,x0,t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t] )

        epsilon = torch.randn_like(x0) 

        return sqrt_alpha_hat * x0 + sqrt_one_minus_alpha_hat *epsilon, epsilon 
    
    

        


