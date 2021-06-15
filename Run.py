import Network.KwaveWrapper.Utilities as u
from Trainer import Trainer
from Network.POCSNet import POCSNet
from DataLogger.DataLogger import DataLogger
from Data.DataLoader import PACTDataset

import torch

device = "cuda:2"
doTest = True
NR_OF_EPOCHS = 15

startItter = -1
itters = 5
NNgrid_Nx = 856
NNgrid_FOV = [-75e-3, 75e-3]
Kgrid_Nx = 2048
Kgrid_FOV = [-111e-3, 111e-3]
Nt = 4000
dt = 25e-9

data_path = r"Data\GreedyDataGT.h5"

full_sensor_params = {
    "layout": "full_circular",
    "phi": 3.14,
    "R": 110e-3,
    "start": 0,
    "end": -2*3.14
}
S = u.SensorArray(512, params = full_sensor_params)
NNgrid = u.Grid(NNgrid_Nx, NNgrid_FOV, NNgrid_Nx, NNgrid_FOV, Nt, dt)
Kgrid = u.Grid(Kgrid_Nx, Kgrid_FOV, Kgrid_Nx, Kgrid_FOV, Nt, dt)
S_real, S_art = u.splitArray(S, 128)

data = PACTDataset(data_path, S_real, S_art)
network = POCSNet(S_real, S_art, Kgrid, NNgrid, itters, device).to(device)

dl = DataLogger(type="test")
criterion = {
    "optimizer": torch.optim.Adam(network.parameters()),
    "criterion": torch.nn.MSELoss()   
}

trainer = Trainer(data, criterion, dl,device, doTest=doTest)

if startItter == -1:
    print("Processing First Step")
    trainer.firstStep(network)

for ii in range(itters):
    print("============TRAINING ITTERATION %i===============" % ii)
    for epoch in range(NR_of_EPOCHS):
        print("============EPOCH %i==========================" % ii)
        print("Training Step...")
        trainer.step(network, ii, 'train')

        print("Testing Step...")
        trainer.step(network, ii, 'test')
        
        print("Validation Step...")
        trainer.step(network, ii, 'val')

    trainer.saveStep(self, network, ii)
    torch.save(network.load_state_dict(), "NetworkSaves\\network_itter%i.pt" % ii)

        