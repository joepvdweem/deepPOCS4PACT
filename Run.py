import Network.DAS_COLE.Utilities as u
from Trainer import Trainer
from Network.POCSNet_DAS_COLE import POCSNet
from DataLogger.DataLogger import DataLogger
from Data.DataLoader import PACTDataset

import torch

device = "cuda"
doTest = True
NR_OF_EPOCHS = 15

startItter = -1
itters = 5
Nx = 856
FOVx = [-75e-3, 75e-3]
super_grid_Nx = 2048
Nt = 4000
dt = 25e-9
t0 = 0
Ns = 512 #full_array sensor amount
Ns_real = 128 #actual array sensor amount
Ns_interpolation_factor = 2 

data_path = r"Data\Data\placeHolder.h5"

full_sensor_params = {
    "layout": "full_circular",
    "phi": 3.14,
    "R": 110e-3,
    "start": 0,
    "end": -2*3.14
}
S = u.SensorArray(512, params = full_sensor_params)
grid = u.Grid(Nx, FOVx, Nx, FOVx, Nt, dt)
super_grid = u.Grid(super_grid_Nx, FOVx, super_grid_Nx, FOVx, Nt, dt)
S_real, S_art = S.SplitArray(Ns_real)

data = PACTDataset(data_path, S_real, S_art)
network = POCSNet(itters, grid, super_grid, S_real, S_art, Ns_interpolation_factor).to(device)

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
    for epoch in range(NR_OF_EPOCHS):
        print("============EPOCH %i==========================" % ii)
        print("Training Step...")
        trainer.step(network, ii, 'train')

        print("Testing Step...")
        trainer.step(network, ii, 'test')
        
        print("Validation Step...")
        trainer.step(network, ii, 'val')

    trainer.saveStep(network, ii)
    torch.save(network.load_state_dict(), "NetworkSaves\\network_itter%i.pt" % ii)

        