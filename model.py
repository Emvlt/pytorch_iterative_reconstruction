import torch
import torch.nn.functional as F

class Rotation_Network(torch.nn.Module):
    def __init__(self, dimension:int, theta:float, device:torch.device) -> None:
        super(Rotation_Network, self).__init__()
        if dimension ==2:
            self.rotation_tensor = torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                                                 [torch.sin(theta), torch.cos(theta), 0]]).unsqueeze(0)
        elif dimension ==3:
            self.rotation_tensor = torch.tensor([[torch.cos(theta), -torch.sin(theta),0,0],
                                                 [torch.sin(theta), torch.cos(theta), 0,0],
                                                 [0,0,1,0]]).unsqueeze(0)
                                   
        self.device = device

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        grid = F.affine_grid(self.rotation_tensor, x.size()).to(self.device)
        return F.grid_sample(x, grid, padding_mode='zeros')

class PSNRLoss(torch.nn.Module):
    def __init__(self):
        super(PSNRLoss, self).__init__()
        self.MSE = torch.nn.MSELoss()
    def forward(self, tensorA:torch.Tensor,tensorB:torch.Tensor) -> torch.Tensor:
        return 20*(torch.log10(torch.max(tensorB)))-10*torch.log10(self.MSE(tensorA,  tensorB))
