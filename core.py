
from typing import Dict
import torch
from tqdm import tqdm
import pathlib
from model import Rotation_Network
from utils import get_projection_tensor
from torch.utils.tensorboard import SummaryWriter

def sample(dimension:int, angles:torch.Tensor, volume:torch.Tensor, grid_properties:Dict, batch_size:int, detector_properties:Dict, device:torch.device, save_path:pathlib.Path):
    if not save_path.is_file():
        projection_tensor = get_projection_tensor(grid_properties, batch_size, detector_properties['beam_geometry']).to(device)

        rotation_tensor_dict  = {f'{i}': Rotation_Network(theta, device) for i, theta in enumerate(angles)}

        dim = 3 if dimension==2 else 4

        sinogram = torch.zeros((batch_size,len(angles),detector_properties['height'],detector_properties['width']), device=device)

        for angle_index in tqdm(range(len(angles))):
            sinogram[:,angle_index] = torch.sum(rotation_tensor_dict[f'{angle_index}'](volume).mul(projection_tensor), dim=dim)

        torch.save(sinogram, save_path)

def reconstruct(dimension:int, angles:torch.Tensor, projections:torch.Tensor, grid_properties:Dict, batch_size:int, beam_geometry:str, device:torch.device):
    projection_tensor = get_projection_tensor(grid_properties, batch_size, beam_geometry).to(device)
    rotation_tensor_dict  = {f'{i}': Rotation_Network(theta, device) for i, theta in enumerate(angles)}

    n_steps = 200

    mse_loss = torch.nn.MSELoss()

    reconstruction = torch.zeros((batch_size,grid_properties['depth'],grid_properties['height'],grid_properties['width']), requires_grad=True, device=device)

    optimiser  = torch.optim.Adam([reconstruction], lr=1e-4)

    dim = 3 if dimension==2 else 4

    writer = SummaryWriter()

    for n in tqdm(range(n_steps)):
        sinogram_loss = 0
        for angle_index in range(len(angles)):
            optimiser.zero_grad()
            target_acquisition  = projections[:,angle_index]
            infered_acquisition = torch.sum(rotation_tensor_dict[f'{angle_index}'](reconstruction).mul(projection_tensor), dim=dim)
            loss = mse_loss(infered_acquisition,target_acquisition)
            sinogram_loss += loss.item()
            loss.backward()
            optimiser.step()
        
        writer.add_scalar('Sinogram loss', sinogram_loss, n)

