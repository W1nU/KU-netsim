import torch
import torch.nn as nn

class AE_(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.enc = nn.Sequential(
            nn.Conv2d(self.cfg.in_ch, self.cfg.out_ch, (self.cfg.f_size, self.cfg.f_size), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.cfg.out_ch, self.cfg.out_ch, (self.cfg.f_size, self.cfg.f_size), stride=1, padding=1),
            nn.Sigmoid(),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(self.cfg.out_ch, self.cfg.in_ch, (self.cfg.f_size + 1, self.cfg.f_size + 1), stride=2, padding=1),
            #nn.Tanh(),
            nn.Sigmoid(),
        )
        self.opt = torch.optim.Adam(self.parameters(), lr=0.0002, betas=(0.9, 0.999))

    def encode(self, x):
        return self.enc(x)
    
    def decode(self, h):
        return self.dec(h)
    
    def forward(self, x):
        return self.dec(self.enc(x))
    
class AE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if self.cfg.lv == 1:
            self.enc = nn.Sequential(
                nn.Conv2d(self.cfg.in_ch, self.cfg.out_ch * 4, (3, 3), stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(self.cfg.out_ch * 4, self.cfg.out_ch, (3, 3), stride=2, padding=1),
                nn.ReLU(),
            )
            self.dec = nn.Sequential(
                nn.ConvTranspose2d(self.cfg.out_ch, self.cfg.out_ch, (self.cfg.f_size + 1, self.cfg.f_size + 1), stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(self.cfg.out_ch, self.cfg.in_ch, (self.cfg.f_size + 1, self.cfg.f_size + 1), stride=2, padding=1),
                nn.Tanh(),
            )
        else:
            self.enc = nn.Sequential(
                nn.Conv2d(self.cfg.in_ch, self.cfg.out_ch, (self.cfg.f_size, self.cfg.f_size), stride=2, padding=1),
                nn.ReLU(),
            )
            self.dec = nn.Sequential(
                nn.ConvTranspose2d(self.cfg.out_ch, self.cfg.in_ch, (self.cfg.f_size + 1, self.cfg.f_size + 1), stride=2, padding=1),
                nn.ReLU(),
            )
        self.opt = torch.optim.Adam(self.parameters(), lr=0.0002, betas=(0.9, 0.999))

    def encode(self, x):
        return self.enc(x)
    
    def decode(self, h):
        return self.dec(h)
    
    def forward(self, x):
        return self.dec(self.enc(x))