import torch
import numpy as np

#god parameters
NT_RECONSTRUCT_ERROR = 4; #extra time points eaten by the reconstruction methods

class Grid(torch.nn.Module):
    
    def __init__(self, Nx, xrange, Ny, yrange, Nt, dt, Nz=0, zrange=0, c=1500):
        super(Grid, self).__init__()

        self.Nx = Nx
        self.xrange = xrange
        self.x  = torch.linspace(xrange[0], xrange[1], Nx)
        self.dx = abs(self.xrange[0] - self.xrange[1]) / self.Nx
        
        self.Ny = Ny
        self.yrange = yrange
        self.y  = torch.linspace(yrange[0], yrange[1], Ny)
        self.dy = abs(self.yrange[0] - self.yrange[1]) / self.Ny
        
        self.Nz = Nz
        self.zrange = zrange
        if self.Nz != 0:
            self.z = torch.linspace(zrange[0], zrange[1], Nz)
            self.dz = abs(self.zrange[0] - self.zrange[1]) / self.Nz
        else:
            self.z = torch.tensor(0.);
            self.dz = 0
            
        self.Nt = Nt
        self.dt = dt
        self.t = torch.arange(0, Nt)*dt
        self.t_max = (Nt - NT_RECONSTRUCT_ERROR)*dt;
        self.t_min = 0;
        
        self.c = c
            
        self.x = torch.nn.Parameter(self.x, requires_grad = False)
        self.y = torch.nn.Parameter(self.y, requires_grad = False)
        self.z = torch.nn.Parameter(self.z, requires_grad = False)
        self.t = torch.nn.Parameter(self.t, requires_grad = False)
        
      
        
    def getGridAxis(self):
        return (self.x, self.y, self.z)
        
    def getFullGrid(self, dims = 2):
        if dims == 2:
            xx, yy = torch.meshgrid(self.x, self.y)
            return torch.stack([xx, yy], -1)
        else:
            pass
        
    def getDims(self):
        if self.Nz == 0:
            return (self.Nx, self.Ny, 1)
        else:
            return (self.Nx, self.Ny, self.Nz)
    
    def getRes(self):
        if self.Nz == 0:
            return (self.dx, self.dy, 1e-4)
        else:
            return (self.dx, self.dy, self.dz)

std_sensor_params = {
    "layout": "full_circular",
    "phi": 0,
    "R": 110e-3}

class SensorArray(torch.nn.Module):
    def __init__(self, Ns, params = std_sensor_params):
        super(SensorArray, self).__init__()
        self.Ns = Ns
        
        if params["layout"] == "full_circular":
            phi = torch.arange(0, 2*3.141592, 2*3.141592/self.Ns) + params["phi"]
            self.sensor_pos = torch.stack([params["R"] * torch.cos(phi), params["R"] * torch.sin(phi)], -1)
        elif params["layout"] == "partial_circular":
            phi = torch.linspace(params["phi_start"], params["phi_end"], Ns)
            self.sensor_pos = torch.stack([params["R"] * torch.cos(phi), params["R"] * torch.sin(phi)], -1)
        elif params["layout"] == "split":
            self.sensor_pos = params["sensor_pos"]
            self.Ns = len(self.sensor_pos)
            self.idx = params["idx"]
        
        else:
            print("invalid_layout")
        self.sensor_pos = torch.nn.Parameter(self.sensor_pos, requires_grad = False)
            
        self.mask_type = 0
        
        # self.waveform = torch.tensor(waveform, requires_grad = False, dtype=torch.float32)
        # self.waveform = torch.nn.Parameter(self.waveform, requires_grad = False)
    
    def getSensorArray(self):
        return self.sensor_pos
    
    def getMask(self, grid):
        
        res = np.array(grid.getRes()[:2])
        
        sensor_pos = torch.round(self.sensor_pos / res)
        sensor_pos[:, 0] = sensor_pos[:, 0] + grid.Nx//2
        sensor_pos[:, 1] = sensor_pos[:, 1] + grid.Ny//2
        
        sensor_mask = torch.zeros(grid.getDims())
        sensor_mask[sensor_pos[:, 0].long(), sensor_pos[:, 1].long()] = 1
        
        sensor_idx = torch.nonzero(sensor_mask.view(-1))
        
        return (sensor_mask.detach().cpu().numpy(), sensor_idx.detach().cpu().numpy())
    
    def getWaveform(self):
        return self.waveform
    
    def getWaveformShape(self):
        return self.waveform.size()

    def getIdx(self):
        return self.idx

class Scaler(torch.nn.Module):
    def __init__(self, Kgrid, NNgrid):
        super(Scaler, self).__init__()
        self.Kgrid = Kgrid
        self.NNgrid = NNgrid
        self.subGridNx = np.round(Kgrid.Nx / (NNgrid.dx * 2)) * 2 # round to nearest even number
        self.padAmount = (self.subGridNx - NNgrid.Nx)

    def scaleForward(self, x):
        x = torch.pad(x, [self.padAmount, self.padAmount]).unsqueeze(1) # adds zeros to match the FOV of the other grid
        x = torch.nn.functional.interpolate(x, size = [self.Kgrid.Nx , self.Kgrid.Nx], mode="bicubic")
        return x

    def scaleBackward(self, x):
        x = torch.nn.functional.interpolate(x, size = [self.subGridNx , self.subGridNx], mode="bicubic")
        x = x[:,padAmount : (subGridNx - padAmount),  padAmount : (subGridNx - padAmount)] #un-pad the image
        return x

def splitArray(S, Ns):
    sensor_array = S.getSensorArray()
    S_real_params = {
        "sensor_pos": sensor_array[:Ns],
        "layout": "split",
        "idx": np.arange(0, Ns)
    }
    S_art_params = S_real_params.copy()
    S_art_params["sensor_pos"] = sensor_array[Ns:]
    S_art_params["idx"] =  np.arange(Ns, len(sensor_array))

    S_real = SensorArray(0, params=S_real_params)
    S_art = SensorArray(0, params=S_art_params)

    return S_real, S_art
    
if __name__ == '__main__':
    S = SensorArray(256)
    g = Grid(512, [-111e-3, 111e-3], 512, [-111e-3, 111e-3], 4000, 25e-9)
    a = S.getMask(g)

    S_real, S_art = splitArray(S, 64)
    
    a1 = S_real.getMask(g)
    print(a)

    import matplotlib.pyplot as plt
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
        
    plt.imshow(np.squeeze(a1[0]))

    plt.show()


        
        
        
        
        
    