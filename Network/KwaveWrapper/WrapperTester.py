from Wrapper import kWaveWrapper
import Utilities as u
import scipy.io as sio
import numpy as np
import torch

full_sensor_params = {
    "layout": "full_circular",
    "phi": 3.14,
    "R": 110e-3,
    "start": 0,
    "end": -2*3.14
}

S = u.SensorArray(512, params = full_sensor_params)
NNg = u.Grid(856, [-111e-3, 111e-3], 856, [-111e-3, 111e-3], 4000, 25e-9)
kwg = u.Grid(2048, [-111e-3, 111e-3], 2048, [-111e-3, 111e-3], 4000, 25e-9)
a = S.getMask(kwg)

Kww = kWaveWrapper(S, kwg, device="cuda:2")

input_data = sio.loadmat("input_image.mat", squeeze_me=True)["data"]
input_data = np.expand_dims(input_data, 0)
input_data = torch.tensor(input_data)

test_sensor_data = Kww.calculateForward(input_data)
print(test_sensor_data.size())
test_image = Kww.calculateBackward(test_sensor_data)
print(test_sensor_data.size())

sio.savemat("test.mat", {"sensor_data": test_sensor_data.numpy(), "image": test_image.numpy()})
