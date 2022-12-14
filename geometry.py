import torch 
import pathlib
from dataclasses import dataclass
import json
from typing import Union, List
import math 
import time


#     ^ Z
#     |
#     |___ 
#    /     Y
#   / X
import matplotlib.pyplot as plt

def g_i_v(float_value):
    t = float_value%1
    return math.floor(float_value) if t<=0.5 else math.ceil(float_value)

@dataclass
class Geometry():
    def __init__(self) -> None:
        self.DSD:float = None
        self.DSO:float = None 
        # NOT TESTED FOR:
        #   RECTANGULAR VOLUMES
        #   RECTANGULAR VOXELS
        #   DETECTOR WITH OFFSET 
        #   RECTANGULAR DETECTOR (3D)
        #   RECTANGULAR PIXELS IN THE DETECTOR

        ## VOLUME PROPERTIES 
        self.n_voxels:List(int)   = None # Number of voxels in the volume (z,x,y) [no unit]
        self.s_voxels:List(float) = None # Size of the voxels                     [mm]

        ## DETECTOR PROPERTIES
        self.n_pixels_detector:List(int)   = None # Number of voxels in the volume (x,y,z) [no unit]
        self.s_pixels_detector:List(float) = None # Size of the voxels                     [mm]

        self.dimension:int = None
        self.beam_geometry:str = None
        self.projection_tensor_path:str = None

        self.sparse:bool = None

    def read_from_dict(self, dict):
        try:
            self.DSD = dict['DSD']
            self.DSO = dict['DS0']

            self.n_voxels = dict['n_voxels']
            self.s_voxels = dict['s_voxels']
            self.n_voxels_x_y_z = [self.n_voxels[1],self.n_voxels[2],self.n_voxels[0]]
            self.s_voxels_x_y_z = [self.s_voxels[1],self.s_voxels[2],self.s_voxels[0]]

            self.n_pixels_detector = dict['n_pixels_detector']
            self.s_pixels_detector = dict['s_pixels_detector']

            self.beam_geometry = dict['beam_geometry']
            self.dimension = dict['dimension']
            self.projection_tensor_path = dict['projection_tensor_path']

            self.sparse = dict['sparse']

        except KeyError as error:    
            print(error.args)

    def read_from_json(self, path_to_json:Union[pathlib.Path, str]):
        self.read_from_dict(json.load(open(path_to_json)))

    @property
    def get_volume_dimensions(self):
        if self.n_voxels is not None and self.s_voxels is not None:
            self.volume_dimensions = []
            for s, n in zip(self.s_voxels, self.n_voxels):
                self.volume_dimensions.append(s*n)  # [mm]

    @property
    def get_pixel_values(self):
        self.DSD_pix = g_i_v(self.DSD/self.s_voxels[2])
        self.DSO_pix = g_i_v(self.DSO/self.s_voxels[2])
        self.ratio_dim_x = self.s_pixels_detector[1]/self.s_voxels[1]
        self.ratio_dim_z = self.s_pixels_detector[0]/self.s_voxels[0]
        self.detector_pix_dimensions = [
            g_i_v(self.n_pixels_detector[1]*self.ratio_dim_x), 
            g_i_v(self.n_pixels_detector[0]*self.ratio_dim_z)
            ]
        self.source_position = [
            g_i_v(self.detector_pix_dimensions[0]/2),
            0,           
            g_i_v(self.detector_pix_dimensions[1]/2)
            ]
        self.detector_center = [
            g_i_v(self.detector_pix_dimensions[0]/2),
            self.DSD_pix,
            g_i_v(self.detector_pix_dimensions[1]/2)
            ]

    @property
    def get_detector_centers(self):
        self.bin_centers_pix = [[] for i in range(g_i_v(self.n_pixels_detector[0]*self.ratio_dim_z))]
        self.n_bins = self.detector_pix_dimensions[0]*self.detector_pix_dimensions[1]
        for row in range(self.detector_pix_dimensions[1]):
            for col in range(self.detector_pix_dimensions[0]):
                self.bin_centers_pix[row].append([col,self.DSD_pix,row])
            
    def initialise(self):
        self.get_volume_dimensions
        self.get_pixel_values
        self.get_detector_centers
        
    def compute_sparse_projection_tensor(self):
        c_indices = []
        x_indices = []
        y_indices = []
        values  = []
        t0 = time.time()
        for row_index in range(len(self.bin_centers_pix)): 
            for col_index, bin_position in enumerate(self.bin_centers_pix[row_index]):
                channel_index = row_index*len(self.bin_centers_pix[row_index])+col_index
                if self.beam_geometry == 'cone':
                    v_0 = self.source_position
                elif self.beam_geometry == 'parallel':
                    v_0 = [bin_position[0],0,bin_position[2]]
                c_, x_, y_, vals = self.sparse_grid_intersection( v_0, bin_position, channel_index)
                c_indices += c_
                x_indices += x_
                y_indices += y_
                values += vals
        print(f'Elapsed time : {time.time()-t0}')
        torch.save(torch.sparse_coo_tensor([c_indices,x_indices,y_indices], values), self.projection_tensor_path)
        
    def compute_dense_projection_tensor(self):
        if self.dimension == 2:
            projection_tensor:torch.Tensor = torch.zeros((self.n_bins, self.n_voxels[1], self.n_voxels[2]))
        else:
            projection_tensor:torch.Tensor = torch.zeros((self.n_bins, self.n_voxels[0], self.n_voxels[1], self.n_voxels[2]))
        t0 = time.time()
        for row_index in range(len(self.bin_centers_pix)): 
            for col_index, bin_position in enumerate(self.bin_centers_pix[row_index]):
                channel_index = row_index*len(self.bin_centers_pix[row_index])+col_index
                if self.beam_geometry == 'cone':
                    v_0 = self.source_position
                elif self.beam_geometry == 'parallel':
                    v_0 = [bin_position[0],0,bin_position[2]]
                projection_tensor[channel_index] = self.dense_grid_intersection(v_0, bin_position)
                
        print(f'Elapsed time : {time.time()-t0}')
        print(f'Projection Tensor Size : {projection_tensor.size()}')
        torch.save(projection_tensor, self.projection_tensor_path)
        
    def compute_projection_tensor(self):
        if self.sparse:
            self.compute_sparse_projection_tensor()
        else:
            self.compute_dense_projection_tensor()

    def dense_straight_2d(self, v_0):
        mat = torch.zeros((self.n_voxels_x_y_z[1], self.n_voxels_x_y_z[0]))
        mat[:,v_0[0] - self.source_position[0] + int(self.n_voxels_x_y_z[0]/2)] = 1
        return mat/self.n_voxels_x_y_z[0]

    def sparse_straight_2d(self, v_0, channel_index):
        l = self.n_voxels_x_y_z[0]
        indx = [[channel_index, v_0[0], i] for i in range(l)]
        vals  = [i/l for i in range(l)]
        return indx, vals

    def dense_siddon_2d(self, v_0:List, v_1:List, dims:List):
        ## WARNING, Y dim HAS to be on first index in dims list
        ## Warning, I think it does not work properly, there seems to be a resolution problem in the sinogram
        N = [self.n_voxels_x_y_z[i] for i in dims]
        Nu, Nv = N[0], N[1]
        R = [v_1[i]-v_0[i] for i in dims]
        Rv = R[1]
        mat = torch.zeros((Nu, Nv))
        offset_u = self.DSO_pix
        offset_v = self.source_position[0] - int(Nv/2)
        a = Rv/self.DSD_pix
        b = self.source_position[0]
        def f(x):
            return a*x+b

        def populate_mat(mat, i):
            y0 = f(i+offset_u)
            y1 = f(i+offset_u+1)
            if y1 < offset_v or offset_v+Nv-1 < y1:
                return mat, False            
            if math.floor(y1)==math.floor(y0):
                mat[i, math.floor(y0)-offset_v] = math.sqrt(1+(y1-y0)**2)
            else:
                def g(y):                    
                    return (y-y0)/(y1-y0)
                if 0<Rv:
                    mat[i, math.floor(y0)-offset_v] = math.sqrt((math.ceil(y0)-y0)**2+(g(math.ceil(y0)) - g(y0))**2)
                    mat[i, math.floor(y1)-offset_v] = math.sqrt((y1-math.floor(y1))**2+(g(y1)-g(math.floor(y1)))**2)
                    for j in range(math.ceil(y0), math.floor(y1)):
                        mat[i, j-offset_v] = math.sqrt(1+(g(j+1)-g(j))**2)
                else:
                    mat[i, math.floor(y0)-offset_v] = math.sqrt((math.floor(y0)-y0)**2+(g(math.floor(y0)) - g(y0))**2)
                    mat[i, math.floor(y1)-offset_v] = math.sqrt((y1-math.ceil(y1))**2+(g(y1)-g(math.ceil(y1)))**2)
                    for j in range(math.floor(y0), math.ceil(y1)):
                        mat[i, j-offset_v] = math.sqrt(1+(g(j+1)-g(j))**2)
                    
            return mat, True
        compute = True
        i = 0
        while compute and i<Nu:
            mat, compute = populate_mat(mat, i)
            i += 1
        return mat/i

    
    
    def dense_grid_intersection(self, v_0:List, v_1:List):
        """
        Computes the intersection of a grid and a path between two voxels
        Args:
            v_0 (List): start_voxel
            v_1 (List): end_voxel
        """
        ## INITIALISATION
        # Get voxel coordinates in a more convenient form
        x1, x2 = v_0[0], v_1[0]
        z1, z2 = v_0[2], v_1[2]
        if x1 == x2 and z1 == z2:
            # Straight ray situation:
            return self.dense_straight_2d(v_0)
        elif x1 == x2 and z1 != z2:
            # Compute siddon in yz plane
            return self.dense_siddon_2d(v_0, v_1, [1,2])
        elif x1 != x2 and z1 == z2:
            # Compute siddon in xy plane
            return self.dense_siddon_2d(v_0, v_1, [1,0])
        else:
            raise NotImplementedError

    def sparse_straight_2d(self, v_0, channel_index):
        l = self.n_voxels_x_y_z[0]
        c_, x_, y_ = [channel_index for i in range(l)], [v_0[0] for i in range(l)], [i for i in range(l)]
        vals  = [i/l for i in range(l)]
        return c_, x_, y_, vals

    def sparse_siddon_2d(self, v_0:List, v_1:List, dims:List, channel_index:int):
        ## WARNING, Y dim HAS to be on first index in dims list
        ## Warning, I think it does not work properly, there seems to be a resolution problem in the sinogram
        N = [self.n_voxels_x_y_z[i] for i in dims]
        Nu, Nv = N[0], N[1]
        R = [v_1[i]-v_0[i] for i in dims]
        Rv = R[1]
        c_, x_, y_ = [], [], []
        vals = []
        offset_u = self.DSO_pix
        offset_v = self.source_position[0] - int(Nv/2)
        a = Rv/self.DSD_pix
        b = self.source_position[0]
        def f(x):
            return a*x+b

        def populate_mat(c_:List, x_:List, y_:List, vals:List, i):
            y0 = f(i+offset_u)
            y1 = f(i+offset_u+1)
            if y1 < offset_v or offset_v+Nv-1 < y1:
                return c_, x_, y_, vals, False            
            if math.floor(y1)==math.floor(y0):
                c_.append(channel_index)
                x_.append(i)
                y_.append(math.floor(y0)-offset_v)
                vals.append(math.sqrt(1+(y1-y0)**2))

            else:
                def g(y):                    
                    return (y-y0)/(y1-y0)
                if 0<Rv:               
                    c_.append(channel_index)
                    x_.append(i)
                    y_.append(math.floor(y0)-offset_v)
                    vals.append(math.sqrt((math.ceil(y0)-y0)**2+(g(math.ceil(y0)) - g(y0))**2))
                    c_.append(channel_index)
                    x_.append(i)
                    y_.append(math.floor(y1)-offset_v)
                    vals.append(math.sqrt((y1-math.floor(y1))**2+(g(y1)-g(math.floor(y1)))**2))

                    for j in range(math.ceil(y0), math.floor(y1)):                        
                        c_.append(channel_index)
                        x_.append(i)
                        y_.append(j-offset_v)
                        vals.append(math.sqrt(1+(g(j+1)-g(j))**2))
                else:
                    c_.append(channel_index)
                    x_.append(i)
                    y_.append(math.floor(y0)-offset_v)
                    vals.append(math.sqrt((math.floor(y0)-y0)**2+(g(math.floor(y0)) - g(y0))**2))        
                    c_.append(channel_index)
                    x_.append(i)
                    y_.append(math.floor(y1)-offset_v)      
                    vals.append(math.sqrt((y1-math.ceil(y1))**2+(g(y1)-g(math.ceil(y1)))**2))

                    for j in range(math.floor(y0), math.ceil(y1)):
                        c_.append(channel_index)
                        x_.append(i)
                        y_.append(j-offset_v)
                        vals.append(math.sqrt(1+(g(j+1)-g(j))**2))
                    
            return c_, x_, y_, vals, True
        compute = True
        i = 0
        while compute and i<Nu:
            c_, x_, y_, vals, compute = populate_mat(c_, x_, y_, vals, i)
            i += 1
        return c_, x_, y_, [val/i for val in vals]

    def sparse_grid_intersection(self, v_0:List, v_1:List, channel_index:int):
        """
        Computes the intersection of a grid and a path between two voxels
        Args:
            v_0 (List): start_voxel
            v_1 (List): end_voxel
        """
        ## INITIALISATION
        # Get voxel coordinates in a more convenient form
        x1, x2 = v_0[0], v_1[0]
        z1, z2 = v_0[2], v_1[2]
        if x1 == x2 and z1 == z2:
            # Straight ray situation:
            return self.sparse_straight_2d(v_0, channel_index)
        elif x1 == x2 and z1 != z2:
            # Compute siddon in yz plane
            return self.sparse_siddon_2d(v_0, v_1, [1,2], channel_index)
        elif x1 != x2 and z1 == z2:
            # Compute siddon in xy plane
            return self.sparse_siddon_2d( v_0, v_1, [1,0], channel_index)
        else:
            raise NotImplementedError