import torch
import numpy as np
import cv2
import pathlib

def normalise(t:torch.Tensor):
    return (t-t.min())/(t.max()-t.min()) 
        
def dicom_to_tensor(load_path:pathlib.Path, save_path:pathlib.Path):
    print(f'Saving {load_path.stem} as a {save_path.suffix} file')
    volume = torch.from_numpy(np.float32(np.load(load_path))).float()
    torch.save(volume.unsqueeze(0).unsqueeze(0), save_path)
       
def dicom_to_image(dimension:int, load_path:pathlib.Path, save_path:pathlib.Path):
    print(f'Saving {load_path.stem} as a jpg file at location {save_path}')
    if dimension == 2:
        volume = np.uint8(normalise(torch.from_numpy(np.float32(np.load(load_path))).float())*255)
        cv2.imwrite(str(save_path), volume)
    else:
        raise NotImplementedError (f'Image writing not implemented for dimension {dimension}')

def tensor_to_image(dimension:int, tensor:torch.Tensor, save_path:pathlib.Path):
    print(f'Saving Tensor as a jpg file at location {save_path}')
    if dimension == 2:
        volume = np.uint8(normalise(tensor[0]).detach().cpu()*255)
        cv2.imwrite(str(save_path), volume)
    else:
        raise NotImplementedError (f'Image writing not implemented for dimension {dimension}')