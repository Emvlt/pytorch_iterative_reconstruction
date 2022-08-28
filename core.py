
from typing import Dict
import torch
from tqdm import tqdm
import pathlib
from model import Rotation_Network
from utils import get_projection_tensor, normalise
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import cv2
import numpy as np

def sample(dimension:int, angles:torch.Tensor, volume:torch.Tensor, grid_properties:Dict, batch_size:int, detector_properties:Dict, device:torch.device, save_path:pathlib.Path, verbose = False):
    if not save_path.is_file():
        projection_tensor = get_projection_tensor(grid_properties, batch_size, detector_properties['beam_geometry']).to(device)

        rotation_tensor_dict  = {f'{i}': Rotation_Network(dimension, theta, device) for i, theta in enumerate(angles)}

        sinogram = torch.zeros((batch_size,len(angles),detector_properties['height'],detector_properties['width']), device=device)

        for angle_index in tqdm(range(len(angles))):
            acquisition = torch.sum(rotation_tensor_dict[f'{angle_index}'](volume).mul(projection_tensor), dim=3)
            sinogram[:,angle_index] = acquisition
            if verbose:
                if dimension == 2:
                    plt.plot(acquisition[0].detach().cpu())
                elif dimension == 3:
                    plt.matshow(acquisition[0,0].detach().cpu())
                plt.show()

        torch.save(sinogram, save_path)

def reconstruct(dimension:int, angles:torch.Tensor, projections:torch.Tensor, grid_properties:Dict, batch_size:int, beam_geometry:str, device:torch.device, save_path:pathlib.Path, verbose = False):
    projection_tensor = get_projection_tensor(grid_properties, batch_size, beam_geometry).to(device)
    rotation_tensor_dict  = {f'{i}': Rotation_Network(dimension, theta, device) for i, theta in enumerate(angles)}

    n_steps = 10

    mse_loss = torch.nn.MSELoss()

    reconstruction = torch.zeros((batch_size,grid_properties['depth'],grid_properties['height'],grid_properties['width']), requires_grad=True, device=device)

    optimiser  = torch.optim.Adam([reconstruction], lr=1e-4)

    writer = SummaryWriter()

    for n in tqdm(range(n_steps)):
        sinogram_loss = 0
        for angle_index in tqdm(range(len(angles))):
            optimiser.zero_grad()
            target_acquisition  = projections[:,angle_index]
            infered_acquisition = torch.sum(rotation_tensor_dict[f'{angle_index}'](reconstruction).mul(projection_tensor), dim=3)
            loss = mse_loss(infered_acquisition,target_acquisition)
            sinogram_loss += loss.item()
            loss.backward()
            optimiser.step()

        if verbose:
            if dimension == 2:
                cv2.imwrite(f'images/reconstruction_{dimension}_{n}.jpg', np.uint8(normalise(reconstruction[0,0]).detach().cpu()*255))                
            elif dimension == 3:
                cv2.imwrite(f'images/reconstruction_{dimension}D_{n}.jpg', np.uint8(normalise(infered_acquisition[0]).detach().cpu()*255))     
        
        writer.add_scalar(f'Sinogram loss {dimension}', sinogram_loss, n)
    if verbose:
        torch.save(reconstruction, save_path)
