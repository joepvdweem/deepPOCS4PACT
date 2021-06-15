import torch
import numpy as np

#god parameters
NT_RECONSTRUCT_ERROR = 4; #extra time points eaten by the reconstruction methods

class Grid(torch.nn.Module):
    
    def __init__(self, Nx, xrange, Ny, yrange, Nt, dt, Nz=0, zrange=0):
        super(Grid, self).__init__()

        self.Nx = Nx
        self.xrange = xrange
        self.x = torch.linspace(xrange[0], xrange[1], Nx)
        
        self.Ny = Ny
        self.yrange = yrange
        self.y = torch.linspace(yrange[0], yrange[1], Ny)
        
        self.Nz = Nz
        self.zrange = zrange
        if self.Nz != 0:
            self.z = torch.linspace(zrange[0], zrange[1], Nz)
        else:
            self.z = torch.tensor(0.)
            
        self.Nt = Nt
        self.dt = dt
        self.t = torch.arange(0, Nt)*dt
        self.t_max = (Nt - NT_RECONSTRUCT_ERROR)*dt
        self.t_min = 0
            
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

std_sensor_params = {
    "layout": "full_circular",
    "phi": 0,
    "R": 110e-3,
    "start": 0,
    "end":2*3.1415
    }

class SensorArray(torch.nn.Module):
    def __init__(self, Ns, params = std_sensor_params):
        super(SensorArray, self).__init__()
        self.t0 = 0
        self.Ns = Ns
        self.layout = params["layout"]
        
        if params["layout"] == "full_circular":
            phi = torch.linspace(0, 2*3.1415927, self.Ns) + params["phi"]
            self.sensor_pos = torch.stack([params["R"] * torch.cos(phi), params["R"] * torch.sin(phi)], -1)
        if params["layout"] == "partial_circular":
            phi = torch.linspace(params["start"], params["end"], self.Ns) + params["phi"]
            self.sensor_pos = torch.stack([params["R"] * torch.cos(phi), params["R"] * torch.sin(phi)], -1)
        if params["layout"] == "subarray":
            self.indexes = params["indexes"]
            self.sensor_pos = params["sensor_pos"]
            self.t0 = params["t0"]
        
        self.sensor_pos = torch.nn.Parameter(self.sensor_pos, requires_grad = False)
            
        
    def getSensorArray(self):
        return self.sensor_pos

    def SplitArray(self, num_of_real):
        indexes = np.arange(0, self.Ns)
        real_idx = np.sort(indexes[:num_of_real])
        art_idx = np.sort(indexes[num_of_real:])

        realArray = SensorArray(num_of_real, {"layout": "subarray",
                    "sensor_pos": self.sensor_pos[real_idx],
                    "indexes": real_idx,
                    "t0": self.t0})
        artArray = SensorArray(self.Ns - num_of_real, {"layout": "subarray",
                    "sensor_pos": self.sensor_pos[art_idx],
                    "indexes": art_idx,
                    "t0": self.t0})

        return realArray, artArray

    def generateRandomSubArray(self, num_of_real):
        indexes = np.arange(0, self.Ns)
        np.random.shuffle(indexes)
        real_idx = np.sort(indexes[:num_of_real])
        art_idx = np.sort(indexes[num_of_real:])

        realArray = SensorArray(num_of_real, {"layout": "subarray",
                    "sensor_pos": self.sensor_pos[real_idx],
                    "indexes": real_idx,
                    "t0": self.t0})
        artArray = SensorArray(self.Ns - num_of_real, {"layout": "subarray",
                    "sensor_pos": self.sensor_pos[art_idx],
                    "indexes": art_idx,
                    "t0": self.t0})

        return realArray, artArray
        
def generateFOVmask(sensors, grid, c):
    # map outer points of sensor
    # make circles around these outer points
    # map parallel curve to sensor shape
    grid_coords = grid.getFullGrid()

    R_range = grid.Nt * grid.dt * c   
    sensor_min = sensors.sensor_pos[0].view(1, 1, -1)
    sensor_max = sensors.sensor_pos[-1].view(1, 1, -1)

    mask = torch.norm(grid_coords - sensor_min, dim=2) < R_range
    mask = mask | (torch.norm(grid_coords - sensor_max, dim=2) < R_range)

    return mask

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    device = 'cuda'

    sensor_args = {
        "layout": "partial_circular",
        "phi": 0,
        "R": 110e-3,
        "start": 0,
        "end": 3.1415/2

    }

    grid = Grid(1024, [-110e-3, 110e-3], 1024, [-110e-3, 110e-3], 500, 100e-9).to(device)
    sensor_array = SensorArray(512, params = sensor_args).to(device)

    mask = generateFOVmask(sensor_array, grid, 1500)

    plt.imshow(mask.double().cpu().squeeze())
    plt.show()
    