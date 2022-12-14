
from typing import Dict
import torch
import pathlib
from geometry import Geometry
from core import sample, reconstruct
from utils import dicom_to_tensor, dicom_to_image
import argparse

def test(dimension:int, geom:Geometry, training_dict:Dict, sampling_dict:Dict, sparsity:str):
    print(f'----------- Beginning {dimension}D test-----------')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    angles = torch.linspace(0, sampling_dict['sampling_amplitude'], sampling_dict['n_acquisitions'])
    
    image_load_path = pathlib.Path(f'test_data/test_phantom_{dimension}D.npy')
    image_save_path = pathlib.Path(f'images/{dimension}D/test_phantom.jpg')
    if not image_save_path.is_file():
        dicom_to_image(dimension, image_load_path, image_save_path)

    tensor_load_path = pathlib.Path(f'test_data/test_phantom_{dimension}D.npy')
    tensor_save_path = pathlib.Path(f'test_data/test_phantom_{dimension}D.pt')
    if not tensor_save_path.is_file():
        dicom_to_tensor(tensor_load_path, tensor_save_path)

    volume = torch.load(f'test_data/test_phantom_{dimension}D.pt').to(device)

    sinogram_save_path = pathlib.Path(f'test_data/{sparsity}_test_sinogram_{dimension}D_{geom.beam_geometry}.pt')

    sample(dimension, angles, volume, geom, training_dict, device, sinogram_save_path, verbose=False)

    projections = torch.load(f'test_data/{sparsity}_test_sinogram_{dimension}D_{geom.beam_geometry}.pt').to(device)

    reconstruction_save_path = pathlib.Path(f'test_data/{sparsity}_reconstructed_volume_{dimension}D_{geom.beam_geometry}.pt')

    reconstruct(dimension, angles, projections, geom, training_dict, device, reconstruction_save_path, True, video=False)

def get_test_parameters(dimension, beam_geometry, sparsity:str):
    geom = Geometry()
    geom.read_from_json(pathlib.Path(f'test_data/{sparsity}_{dimension}D_geometry_dict_{beam_geometry}.json'))

    geom.initialise()

    if not pathlib.Path(geom.projection_tensor_path).is_file():
        geom.compute_projection_tensor()

    training_dict = {
        'batch_size':1,
        'n_steps':80,
        'learning_rate':1e-4
    }
    sampling_dict = {
        'n_acquisitions' : 512,
        'sampling_amplitude' : 2*3.1415927410125732
    }
        
    return geom, training_dict, sampling_dict
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dimension',type=int, default = 2, required=False)
    parser.add_argument('--beam', type=str, required=False, default='cone')
    parser.add_argument('--sparsity', type=str, required=False, default='sparse')

    args = parser.parse_args()
    geometry_properties, training_dict, sampling_dict = get_test_parameters(args.dimension, args.beam, args.sparsity)
    test(args.dimension, geometry_properties, training_dict, sampling_dict, args.sparsity)