
from typing import Dict
import torch
from tqdm import tqdm
import pathlib
from model import Rotation_Network
from utils import normalise,tensor_to_image
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import cv2
import numpy as np
from geometry import Geometry
from skimage.draw import disk
import json
import timeit

def sample(dimension:int, angles:torch.Tensor, volume:torch.Tensor, geom:Geometry, training_dict:Dict, device:torch.device, save_path:pathlib.Path, verbose = False):
    
    if not save_path.is_file():
        print('Beginning Sampling')

        batch_size = training_dict['batch_size']

        projection_tensor:torch.Tensor = torch.load(geom.projection_tensor_path).unsqueeze(0).to(device)
        rotation_tensor_dict  = {f'{i}': Rotation_Network(dimension, theta, device) for i, theta in enumerate(angles)}
        print(projection_tensor.size())

        if geom.dimension == 2:
            sinogram = torch.zeros((batch_size,len(angles),geom.n_bins), device=device)
        else:
            sinogram = torch.zeros((batch_size,len(angles),geom.detector_pix_dimensions[0], geom.detector_pix_dimensions[1]), device=device)

        for angle_index in tqdm(range(len(angles))):
                
            acquisition = torch.sum(rotation_tensor_dict[f'{angle_index}'].forward(volume).mul(projection_tensor), dim=[2,3])

            sinogram[:,angle_index] = acquisition
            if verbose:
                if dimension == 2:
                    plt.plot(acquisition[0].detach().cpu())
                elif dimension == 3:
                    plt.matshow(acquisition[0,0].detach().cpu())
                plt.show()

        print(f'Saving sinogram at {save_path}')
        torch.save(sinogram, save_path)
    sinogram = torch.load(save_path)
    pathlib.Path(f'images/{dimension}D/sinograms').mkdir(parents=True, exist_ok=True)
    if not pathlib.Path(f'images/{dimension}D/sinograms/{save_path.stem}.jpg').is_file():
        tensor_to_image(geom.dimension, torch.load(save_path), pathlib.Path(f'images/{dimension}D/sinograms/{save_path.stem}.jpg'))

def reconstruct(dimension:int, angles:torch.Tensor, projections:torch.Tensor, geom:Geometry,training_dict:Dict, device:torch.device, save_path:pathlib.Path, verbose:bool, video:bool):
    projection_tensor = torch.load(geom.projection_tensor_path).unsqueeze(0).to(device)
    rotation_tensor_dict  = {f'{i}': Rotation_Network(dimension, theta, device) for i, theta in enumerate(angles)}

    loss_function = torch.nn.MSELoss()

    reconstruction = torch.zeros((training_dict['batch_size'],geom.n_voxels[0],geom.n_voxels[1],geom.n_voxels[2]), requires_grad=True, device=device)
    optimiser  = torch.optim.Adam([reconstruction], lr=training_dict['learning_rate'])

    if verbose:
        image_save_path = pathlib.Path(f'images/{dimension}D/reconstructions')
        image_save_path.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir =f'runs/{geom.beam_geometry}_{dimension}D')

    if video:
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        video_writer = cv2.VideoWriter('reconstruction.mp4', fourcc, 32, (512,512),0)

    mask = torch.ones(reconstruction.size(), requires_grad=False, device=device)
    if dimension == 2:
        center = (int(geom.n_voxels[1]/2),int(geom.n_voxels[2]/2))
        radius = int(int(geom.n_voxels[2]/2))
        rr, cc = disk(center, radius)
        mask[:,:,rr,cc] = 0
        mask = 1 - mask

    for n in tqdm(range(training_dict['n_steps'])):
        sinogram_loss = 0
        for angle_index in tqdm(range(len(angles))):
            optimiser.zero_grad()
            target_acquisition  = projections[:,angle_index]
            acquisition = torch.sum(rotation_tensor_dict[f'{angle_index}'].forward(reconstruction*mask).mul(projection_tensor), dim=[2,3])
            loss = loss_function(acquisition,target_acquisition) 
            sinogram_loss += loss.item()
            loss.backward()
            optimiser.step()

        json.dump(torch.cuda.memory_stats(device), open('stats.json', 'w'), indent = 4)
        if video:
            video_writer.write(np.uint8(normalise(reconstruction[0,0]).detach().cpu()*255))

        if verbose:
            writer.add_scalar(f'Sinogram loss', sinogram_loss, n)
            cv2.imwrite(str(image_save_path.joinpath(f'{geom.beam_geometry}.jpg')), np.uint8(normalise(reconstruction[0,0]).detach().cpu()*255))
            if n%10==0:
                cv2.imwrite(str(image_save_path.joinpath(f'{geom.beam_geometry}_{n}.jpg')), np.uint8(normalise(reconstruction[0,0]).detach().cpu()*255))                
        
        print(f'Sinogram Loss {sinogram_loss:.4f}')

    torch.save(reconstruction.detach().cpu(), save_path)
    if video:
        video_writer.release()
