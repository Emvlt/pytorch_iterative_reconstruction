import torch
from typing import Dict
import numpy as np
import cv2
import pathlib

def normalise(t:torch.Tensor):
    return (t-t.min())/(t.max()-t.min()) 

def get_projection_tensor(grid_properties:Dict, batch_size:int, beam_geometry='parallel'):
    if beam_geometry == 'parallel':
        try:
            return torch.ones((batch_size,grid_properties['depth'],grid_properties['height'],grid_properties['width']))/grid_properties['width']
        except KeyError as error:    
            print(error.args)
    else:
        raise NotImplementedError(f'Beam Geometry {beam_geometry} not implemented')
        
def dicom_to_tensor(dimension:int, load_path:pathlib.Path, save_path:pathlib.Path):
    if not save_path.is_file():
        volume = normalise(torch.from_numpy(np.float32(np.load(load_path))).float())
        torch.save(volume.unsqueeze(0).unsqueeze(0), save_path)
       
def dicom_to_image(dimension:int, load_path=pathlib.Path(f'test_data/test_phantom_2D.npy'), save_path=pathlib.Path(f'images/test_phantom_2D.jpg')):
    if dimension == 2:
        
        if not save_path.is_file:
            volume = np.uint8(normalise(torch.from_numpy(np.float32(np.load(load_path))).float())*255)
            cv2.imwrite(save_path, volume)
    else:
        print(f'Image writing not implemented for dimension {dimension}')