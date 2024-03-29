import torch
import torch.nn as nn
import torch.nn.functional as F

from . import KwaveWrapper as kww

from .ProxLayers import UNET

class POCSNet(nn.Module):

    def __init__(self, S_real, S_art, Kgrid, NNgrid, itters, device):
        """
        
        Parameters
        ----------
        S_real : 
        S_art : 
        Kgrid : 
        NNgrid : 
        itters : 
        device : 
        """
        super(POCSNet, self).__init__()
        
        self.S_real = S_real
        self.S_art = S_art
        self.Kgrid = Kgrid
        self.NNgrid = NNgrid
        self.itters = itters

        self.model = kww.kWaveWrapper(S_art, Kgrid, medium=False, device=device)
        self.FirstStepModel = kww.kWaveWrapper(S_real, Kgrid, medium=False, device=device)
        self.scaler = kww.Scaler(Kgrid, NNgrid)

        self.proxLayers = nn.ModuleDict()

        for i in range(itters):
            self.proxLayers["N_%i" % i] = UNET.UNet(1, 1)
        
        self.proxLayers["N_end"] = UNET.UNet(1, 1)

    def forward(self, x):
        """
        runs the entire network (so each itteration)

        Parameters
        ----------
        x : the input data in the sensor domain

        Returns
        -------
        y_n : output of the n'th step

        """
        # get first step
        y_0 = self.firstStep(x)
        y_n = y_0
        for i in range(self.itters):
            #run pocs step
            y_n = self.POCSStep(y_n, i) #Visually feasable step
            
            #data consistency step
            y_n = y_n + y_0 #physically plausible step

        # run last step
        y_n = self.lastStep(y_n)
        
        return y_n

    @torch.enable_grad()
    def proxStep(self, x, itter):
        """
        runs only a proximal step, mainly for training purposes
        Parameters
        ----------
        x : input data in the image domain
        itter : the network used in itteration itter

        Returns
        -------
        x : proximally mapped input data in the image domain
        """
        
        x = self.proxLayers["N_%i" % itter](x)
        return x
    
    @torch.no_grad()
    def filterStep(self, x):
        """
        filters the data according to the filter.
        """
        
        x = x.permute(1, 0, 2)
        x = F.conv1d(x, self.filter, padding=self.filterPadding)
        x = x.permute(1, 0, 2)
        return x

    @torch.no_grad()
    def POCSStep(self, x, itter):
        """
        runs the entire pocs step

        Parameters
        ----------
        x : output of previous layer
        itter : itter number

        Returns
        -------
        y_n : output of current layer

        """
        # N(.)
        x = self.proxStep(x, itter) 

        # \Theta
        x = self.scaler.scaleForward(x)
        x = self.model.calculateForward(x)
        x = self.filterStep(x)

        # \Phi
        x = self.model.calculateBackward(x)
        y_n = self.scaler.scaleBackward(x)

        return y_n

    def firstStep(self, x):
        """
        The first step only, only \Psi^
        """
        x = self.FirstStepModel.calculateBackward(x)
        x = self.scaler.scaleBackward(x)
        return x

    def lastStep(self, x):
        """
        The last step only, basically only N(.)
        """
        x = self.proxLayers["N_end"](x)
        return x






    
    