# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 09:55:35 2021

@author: Joepi
"""

import torch
import torch.nn as nn

#    __  __         _      _        
#   |  \/  |___  __| |_  _| |___ ___
#   | |\/| / _ \/ _` | || | / -_|_-<
#   |_|  |_\___/\__,_|\_,_|_\___/__/
#                                   

params = {
    "c": 1500,
    "t0": 0,
    }

class DasModule(nn.Module):
    def __init__(self,
                 grid,
                 sensor_array,
                 args = params,
                 batch_size_forward = 64,
                 batch_size_backward = 8192*2,
                 scatter_threshold = 0,
                 interp = "linear",
                 ):
        
        super(DasModule, self).__init__()
        
        self.c = args["c"]
        self.t0 = sensor_array.t0
        self.interp = interp
        
        self.grid = grid
        self.sensor_array = sensor_array
        
        self.batch_size_forward = batch_size_forward
        self.batch_size_backward = batch_size_backward
        
        self.output_main = nn.Parameter(torch.zeros((1, self.grid.Nx, self.grid.Ny)), requires_grad=False)
        self.zero = nn.Parameter(torch.zeros(1), requires_grad = False)
        
        self.batch_indexes = torch.arange(0, self.batch_size_forward, dtype=torch.long, requires_grad=False)
        self.batch_steps = torch.arange(0, self.sensor_array.Ns//self.batch_size_forward, dtype=torch.long, requires_grad=False)
        
        self.gridF = nn.Parameter(self.grid.getFullGrid(), requires_grad = False)
        self.gridax = self.grid.getGridAxis()[0:2]
        self.gridax = torch.stack(self.gridax, 0)
        self.sensors = nn.Parameter(self.sensor_array.getSensorArray().view(-1, 1, 1, 2), requires_grad = False)
        
        self.y0 = nn.Parameter(torch.zeros(self.batch_size_forward, self.grid.Nx, self.grid.Ny), requires_grad = False)

        self.scatter_thresh = scatter_threshold
        self.een = nn.Parameter(torch.ones(1, self.sensor_array.Ns), requires_grad = False)
        self.output_main_backward = nn.Parameter(torch.zeros((self.sensor_array.Ns, self.grid.Nt)), requires_grad=False)
        self.batch_indexes_backward = torch.arange(0, self.batch_size_backward, dtype=torch.long, requires_grad=False)
     
    def forward(self, x):
        x = revAdjoint.apply(x, self)
        return x
        
class RevDasModule(DasModule):
    def forward(self, x):
        x = revForward.apply(x, self)
        return x
#    ___          _                       ___             _   _             
#   | _ ) __ _ __| |___ __ _ _ ___ _ __  | __|  _ _ _  __| |_(_)___ _ _  ___
#   | _ \/ _` / _| / / '_ \ '_/ _ \ '_ \ | _| || | ' \/ _|  _| / _ \ ' \(_-<
#   |___/\__,_\__|_\_\ .__/_| \___/ .__/ |_| \_,_|_||_\__|\__|_\___/_||_/__/
#                    |_|          |_|                                       

class revForward(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x, module):
        ctx.params = module

        x = revDas(ctx.params, x)    
            
        return x
        
    @staticmethod      
    def backward(ctx, gx):
        
        #d/dt(gx)
        gx = torch.cat([gx, torch.zeros((1, gx.size(1), 1)).to(gx.device)], 2) - torch.cat([torch.zeros((1, gx.size(1), 1)).to(gx.device), gx], 2)
        
        #Time Reversal
        x = das(ctx.params, gx)
                
        return x, None

class revAdjoint(torch.autograd.Function):
        
    @staticmethod
    def forward(ctx, x, module):
        ctx.params = module
        
        #d/dt(gx)
        # x = torch.cat([x, torch.zeros((1, x.size(1), 1)).to(x.device)], 2) - torch.cat([torch.zeros((1, x.size(1), 1)).to(x.device), x], 2)
        
        #Time Reversal
        x = das(ctx.params, x)

        return x
        
    @staticmethod      
    def backward(ctx, gx):
        # 
        x = revDas(ctx.params, gx)

        return x, None
#    ___                 ___             _   _             
#   | _ ) __ _ ___ ___  | __|  _ _ _  __| |_(_)___ _ _  ___
#   | _ \/ _` (_-</ -_) | _| || | ' \/ _|  _| / _ \ ' \(_-<
#   |___/\__,_/__/\___| |_| \_,_|_||_\__|\__|_\___/_||_/__/
#                                                          

def das(module, x):
    """
    An application of the time reversal operator by using a delay and sum method

    Parameters
    ----------
    module : an object of class DasModule
    x : input data of the shape (1, Ns, NT)
        
    Returns
    -------
    output : the reconstructed image in the form of (1, Nx, Ny)

    """
    output = module.output_main.clone()
    
    for ii in module.batch_steps:
        batch_idx = ii*module.batch_size_forward + module.batch_indexes
                    
        distance = torch.norm(module.gridF - module.sensors[batch_idx], dim=-1)
    
        #calculate the true index
        idx = (distance / module.c + module.t0) / module.grid.dt
        idx = torch.where(idx > module.grid.t_max/module.grid.dt, module.zero, idx)
        idx = torch.where(idx < 0, module.zero, idx)
        idx = idx.view(1, module.batch_size_forward, module.grid.Nx*module.grid.Ny)
    
        signal_batch = x[:, batch_idx]
        #get the rounded indexes
        if module.interp == "nn":
            idx = idx.round()
            y0 = torch.gather(signal_batch,2, idx.long())    
            # making the indexes workable
            output += y0.view(module.batch_size_forward, module.grid.Nx, module.grid.Ny).sum(0) 
        elif module.interp == "linear":
            #get the two indexes surrounding the true idx
            d0 = idx.floor()
            d1 = d0 + 1
            
            #weigh the data based on the offset of the two indexes compared to the true idx
            W0 = (idx - d0)
            W1 = (d1 - idx)
            
            y0 = torch.gather(signal_batch,2, d0.long())     
            y1 = torch.gather(signal_batch,2, d1.long())   
            
            output += (W0*y0 + W1*y1).view(module.batch_size_forward, module.grid.Nx, module.grid.Ny).sum(0) 
    
    return output

def revDas(module, x): 
    """
    An application of the COLE algorithm
    
    Parameters
    ----------
    module : A DasModule object containing all the nessesary data
    x : an image with the shape of (Nx, Ny)
    
    Yields
    ------
    output: the output image based on something
    """
    def scatterify(x):
        """
        batches all the nonzero values of x together since the zero values of x dont actually contribute to the final image
        """
         # finds the nonzero values
        indexes = torch.nonzero((x != module.scatter_thresh), as_tuple=False)
        values = x[indexes[:, 0], indexes[:, 1]] #obtains the values
        
        #yields the values
        l = values.size(0)
        for ndx in range(0, l, module.batch_size_backward):
            yield values[ndx:min(ndx + module.batch_size_backward, l)].view(1, -1), indexes[ndx:min(ndx + module.batch_size_backward, l)]

    x = x.squeeze() #prepare x
    output = module.output_main_backward.clone()  #prepare output (clone is quicker then creating new zero filled array)
    
    #obtain the grid and sensors
    grid = module.gridax
    sensors = module.sensors
    
    #start batched reconstruction of time domain
    for ii, data in enumerate(scatterify(x)):
        #obtains the current data and grid indexes for reconstruction
        pixel_y, batch_idx = data 
        batch_size = batch_idx.size(0)
        
        #gathers the grid coordinates based on the batch indexes
        grid_tmp = torch.gather(grid, 1, batch_idx.T).T

        #obtains the distance between the sensors and the batched grid points
        distance = grid_tmp.unsqueeze(0) - sensors
        distance = torch.norm(distance, dim=-1).view(module.sensor_array.Ns, 1, batch_size)

        #translates these distances to indexes on the time scale
        idx = (distance / module.c) / module.grid.dt
        idx = torch.where(idx > module.grid.t_max/module.grid.dt, module.zero, idx) # make all the indexes 0 if they are out of range
        idx = torch.where(idx < 0, module.zero, idx)
        idx = idx.view(module.sensor_array.Ns, -1)
        
        #interps the data on the required indexes based on linear interpolation or nearest neighbour
        if module.interp == "nn":
            idx = idx.round()
            W = pixel_y*module.een.T
            
        elif module.interp == "linear":
            #get the two indexes surrounding the true idx
            d0 = idx.floor()
            d1 = d0 + 1
            
            #weigh the data based on the offset of the two indexes compared to the true idx
            W0 = pixel_y*(idx - d0)
            W1 = pixel_y*(d1 - idx)
            
            idx = torch.cat([d0, d1], 1)
            W = torch.cat([W0, W1], 1)
            
        #add the data to the reconstructed time image
        output.scatter_add_(1, idx.long(), W)
    
    #outputs in the field of (1, Ns, Nt)
    output = output.view(1,module.sensor_array.Ns , -1)    
    output[:, :, :1] = 0
    return output


if __name__ == "__main__":
    import Utilities as u
    import scipy.io as sio    
    import matplotlib.pyplot as plt
    import os
    import numpy as np
    from PIL import Image
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    
    device = "cuda:1"
    grid = u.Grid(1024, [-75e-3, 75e-3], 1024, [-75e-3, 75e-3], 1000, 100e-9).to(device)
    sensor_array = u.SensorArray(512).to(device)
    
    data =    sio.loadmat('data.mat')['sensor_data']
    data =    np.array(Image.fromarray(data).resize(size=(1024, 1024)))
    data =    torch.tensor(data, device=device, dtype = torch.float32, requires_grad=True).squeeze()
    
    #forward operation test so (Nx, Ny) --> (Ns, Nt)
    rdm = RevDasModule(grid, sensor_array).to(device)
    
    outp = rdm(data)
    
    plt.imshow(outp.detach().squeeze().cpu(), cmap='gray')
    plt.title("forward operation test")
    # plt.show()
    
    #backward of forward operation
    loss = torch.nn.MSELoss()(outp, torch.zeros_like(outp))
    grad = torch.autograd.grad(loss, outp)[0]

    gx = torch.cat([grad, torch((1, grad.size(1), 1)).to(grad.device)], 2) - torch.cat([torch.zeros((1, grad.size(1), 1)).to(grad.device), gx], 2)
    
    grad = revDas(rdm, grad)

    plt.imshow(grad.detach().squeeze().cpu(), cmap='gray')
    plt.title("grad forward")
    plt.show()
    
    #time reversal operation: (Ns, Nt) --> (Nx, Ny)
    fwd = DasModule(grid, sensor_array).to(device)
    
    outp_fwd = fwd(outp)
    
    plt.imshow(outp_fwd.detach().squeeze().cpu(), cmap='gray')
    plt.title("reverse operation test")
    plt.show()
    
    #time reversal backward operation
    loss = torch.nn.MSELoss()(outp_fwd, torch.zeros_like(outp_fwd))
    grad = torch.autograd.grad(loss, outp)[0]
    
    plt.imshow(grad.detach().squeeze().cpu(), cmap='gray')
    plt.title("reverse operation test")
    plt.show()
    
    output = {
        "fwd_output": outp.detach().cpu().squeeze().numpy(),
        "bkw_output": outp_fwd.detach().cpu().squeeze().numpy(),
        "grad": grad.detach().cpu().squeeze().numpy()
        }
    
    sio.savemat("test_outp.mat", output)
    
    
    
    
    
    