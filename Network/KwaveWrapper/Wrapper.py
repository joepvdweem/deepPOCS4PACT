# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 11:37:52 2021

@author: Joepi
"""
import os
import subprocess
import h5py
import numpy as np
if __name__ == "__main__":
    from .kwaveInputFile2D import KWaveInputFile
else:
    from .kwaveInputFile2D import KWaveInputFile

import torch

KwaveDir = r"E:\Joep van de Weem\Matlab Libraries\k-Wave\binaries\\"
runOn = "CUDA" # or "OMP" for cpp



def pytorchDecorator(func):

    def inner(self, x):

        x_temp = x.cpu().numpy()

        x_temp = func(self, x_temp)

        x_temp = torch.tensor(x_temp, device=x.device)

        return x_temp        

    return inner

class kWaveWrapper():
    
    def __init__(self, sensor, grid, medium = False, path = r"G:\Joep van de Weem\KwaveNet\Network\KwaveWrapper\\", device="1"):

        self.device = device
        self.sensor = sensor
        self.grid = grid.cpu()
        self.path = path
        # the input file for kwave
        self.inputFile = KWaveInputFile(grid.getDims(), grid.Nt, grid.getRes(),
                                        grid.dt, grid.c);
        
        self.verbose = 2

        
        if medium:
            self.inputFile.set_medium_properties(medium["rho0"], medium["c0"])
        else:
            self.inputFile.set_medium_properties(1000, 1500)
            
            
    def writeToFile(self, directory):        
        with h5py.File(directory + "/inputFile.h5", 'w') as f:
            self.inputFile.write_to_file(f)
        return 1
    
    def loadFromFile(self, directory, item):
        with h5py.File(directory + "/outputFile.h5", "r") as f:
            output = f[item][:]
        return output

    def runKWave(self, flag = ''):

        if runOn == "CUDA":
            device = self.device[-1]
        
        a = subprocess.Popen("\"" + KwaveDir + "kspaceFirstOrder-CUDA.exe\"" + 
                            " -i \"" + self.path + "/inputFile.h5 \"" +
                            " -o \"" + self.path + "/outputFile.h5\"" +
                            " -g " + device +  
                            " " + flag, stdout=subprocess.PIPE, shell=True)

        while True:
            output = a.stdout.readline()
            if a.poll() is not None:
                break
            if output:
                if self.verbose == 2:
                    print(output.strip())
        output = 0
        return output
    
    @pytorchDecorator
    def calculateBackward(self, Im):
        p = np.flip(Im, 2)
        p = p.transpose(1, 2, 0)

        _, source_index = self.sensor.getMask(self.grid)
        
        # p_source_index = np.flatnonzero(source_index.transpose((0, 1)))
        p_source_index = np.reshape(source_index, (-1, 1, 1))

        self.inputFile.set_p_source(p_source_index, p)
        
        sensor_mask = np.ones(self.grid.getDims())
        sensor_mask = np.reshape(sensor_mask, [-1, 1, 1])
        print(sensor_mask.shape)
        self.inputFile.set_sensor_mask(self.sensor.mask_type, sensor_mask)

        self.writeToFile(self.path)
        self.runKWave(flag = '--p_final')
        outp = self.loadFromFile(self.path, 'p_final')
        outp = outp.transpose(0, 2,1)
        outp = outp / (1*self.grid.c**2)
        self.reset()
        
        return outp
    
    @pytorchDecorator
    def calculateForward(self, p0):
        
        _, sensor_mask = self.sensor.getMask(self.grid)
        
        sensor_mask = np.reshape(sensor_mask, [-1, 1, 1])
        
        self.inputFile.set_sensor_mask(self.sensor.mask_type, sensor_mask)

        self.inputFile.set_p0_source_input(p0)
        
        self.writeToFile(self.path) # write to h5 file
        self.runKWave() # run kwave form the h5 file
        outp = self.loadFromFile(self.path, 'p')
        outp = outp.transpose(0, 2, 1)
        self.reset()

        return outp
    
    def reset(self):
        self.inputFile.clear_source()

if __name__ == '__main__':
    import Utilities as u
    import scipy.io as sio
    
    phantom = sio.loadmat("D:/0. TUE/Graduation/DATA/derenzo.mat")["derenzo"]
    phantom = np.pad(phantom, 764)
    phantom = np.reshape(phantom, [2048, 2048, 1])

    sensor_params = {
        "layout": "partial_circular",
        "phi_start": 3.1415,
        "phi_end": 5,
        "R": 110e-3}

    grid = u.Grid(2048, [-111e-3, 111e-3], 2048, [-111e-3, 111e-3], 6000, 25e-9)
    sensor_array = u.SensorArray(128, params=sensor_params)    
    k = kWaveWrapper(sensor_array, grid)
    
    p0 = phantom    
    p = k.calculateForward(p0)
    p_adj = k.calculateAdjoint(p.transpose(2, 1, 0))

    import matplotlib.pyplot as plt
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    
    plt.imshow(np.squeeze(p_adj).T)

       
    