
import torch
import pathlib

from core import sample, reconstruct
from utils import normalise, dicom_to_tensor, dicom_to_image

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