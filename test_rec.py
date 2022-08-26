
from typing import Dict, List
import torch
from tqdm import tqdm
import pathlib
from model import Rotation_Network
from utils import normalise, get_projection_tensor, dicom_to_tensor, dicom_to_image
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

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    n_acquisitions = 512
    sampling_amplitude = 2*3.1415927410125732
    dimension = 2
    angles = torch.linspace(0,  sampling_amplitude, n_acquisitions)
    grid_properties = {
        'depth':1,
        'height':512,
        'width':512
        }
    detector_properties = {
        'height':1,
        'width':512,
        'beam_geometry':'parallel'
        }
    
    batch_size = 1
    beam_geometry = 'parallel'

    dicom_to_image(pathlib.Path(f'test_data/test_phantom_{dimension}D.npy'))
    dicom_to_tensor(pathlib.Path(f'test_data/test_phantom_{dimension}D.npy'), pathlib.Path(f'test_data/test_phantom_2D.pt'))
    volume = torch.load(f'test_data/test_phantom_{dimension}D.pt').to(device)
    save_path = pathlib.Path(f'test_data/test_phantom_{dimension}D.pt')
    sample(dimension, angles, volume, grid_properties, batch_size, detector_properties, device, save_path)
    projections = normalise(torch.load(f'test_data/test_sinogram_{dimension}D.pt').to(device))
    reconstruct(dimension, angles, projections, grid_properties, batch_size, beam_geometry, device)
    
    
if __name__ == '__main__':
    main()