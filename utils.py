import torch
from typing import Dict
import numpy as np
import cv2
import pathlib
import matplotlib.pyplot as plt
from geometry import Geometry

def normalise(t:torch.Tensor):
    return (t-t.min())/(t.max()-t.min()) 

def get_projection_tensor(geom:Geometry, device:torch.cuda.Device) -> torch.Tensor:
    if geom.beam_geometry == 'parallel':
        if geom.dimension == 2:
            #return torch.ones((geom.n_voxels_x_y_z[1]), device=device)/geom.n_voxels_x_y_z[0]
            return torch.load(geom.projection_tensor_path).unsqueeze(0).to(device)
            
           
        elif geom.dimension == 3:
            return torch.ones((geom.n_voxels_x_y_z[0],geom.n_voxels_x_y_z[1]), device=device)/geom.detector_pix_dimensions[0]
        else:
            raise ValueError(f'Dimension can only be 2 or 3, not {geom.dimension}')
        
    elif geom.beam_geometry == 'cone' :
        if geom.dimension == 2:
            return torch.load(geom.projection_tensor_path).unsqueeze(0).to(device)
        elif geom.dimension == 3:
            raise NotImplementedError(f'Beam Geometry cone not implemented for dimension 3')
        else:
            raise ValueError(f'Dimension can only be 2 or 3, not {geom.dimension}')
        
    else:
        raise NotImplementedError(f'Beam Geometry {geom.beam_geometry} not implemented')
        
def dicom_to_tensor(load_path:pathlib.Path, save_path:pathlib.Path):
    volume = normalise(torch.from_numpy(np.float32(np.load(load_path))).float())
    torch.save(volume.unsqueeze(0).unsqueeze(0), save_path)
       
def dicom_to_image(dimension:int, load_path:pathlib.Path, save_path:pathlib.Path):
    if dimension == 2:
        volume = np.uint8(normalise(torch.from_numpy(np.float32(np.load(load_path))).float())*255)
        cv2.imwrite(str(save_path), volume)
    else:
        raise NotImplementedError (f'Image writing not implemented for dimension {dimension}')

def tensor_to_image(dimension:int, tensor:torch.tensor, save_path:pathlib.Path):
    if dimension == 2:
        volume = np.uint8(normalise(tensor[0]).detach().cpu()*255)
        cv2.imwrite(str(save_path), volume)
    else:
        raise NotImplementedError (f'Image writing not implemented for dimension {dimension}')