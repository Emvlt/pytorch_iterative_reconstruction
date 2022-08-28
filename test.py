
from typing import Dict
import torch
import pathlib

from core import sample, reconstruct
from utils import normalise, dicom_to_tensor, dicom_to_image

def test(dimension:int, grid_properties:Dict, detector_properties:Dict):
    print(f'----------- Beginning {dimension}D test-----------')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    n_acquisitions = 512
    sampling_amplitude = 2*3.1415927410125732
    
    angles = torch.linspace(0,  sampling_amplitude, n_acquisitions)
    
    batch_size = 1
    beam_geometry = 'parallel'

    dicom_to_image(dimension, pathlib.Path(f'test_data/test_phantom_{dimension}D.npy'))
    dicom_to_tensor(dimension, pathlib.Path(f'test_data/test_phantom_{dimension}D.npy'), pathlib.Path(f'test_data/test_phantom_{dimension}D.pt'))
    volume = torch.load(f'test_data/test_phantom_{dimension}D.pt').to(device)
    print(volume.size())
    save_path = pathlib.Path(f'test_data/test_sinogram_{dimension}D.pt')
    sample(dimension, angles, volume, grid_properties, batch_size, detector_properties, device, save_path, True)
    projections = normalise(torch.load(f'test_data/test_sinogram_{dimension}D.pt').to(device))
    save_path = pathlib.Path(f'test_data/reconstructed_volume_{dimension}D.pt')
    reconstruct(dimension, angles, projections, grid_properties, batch_size, beam_geometry, device, projections, True)

def get_test_parameters(dimension):
    if dimension == 2:
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
        
    elif dimension == 3:
        grid_properties = {
        'depth':512,
        'height':512,
        'width':512
            }
        detector_properties = {
        'height':512,
        'width':512,
        'beam_geometry':'parallel'
            }

    return grid_properties, detector_properties
    
if __name__ == '__main__':
    '''dimension = 2
    grid_properties, detector_properties = get_test_parameters(dimension)
    test(dimension, grid_properties, detector_properties)'''
    dimension = 3
    grid_properties, detector_properties = get_test_parameters(dimension)
    test(dimension, grid_properties, detector_properties)