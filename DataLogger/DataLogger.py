import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.tensorboard.writer import SummaryWriter
import numpy as np

import datetime
from markdown import markdown

class DataLogger:

    def __init__(self, type="pretrain"):
        self.type = type

        now = datetime.datetime.now()
        self.writer = SummaryWriter(log_dir="logs\\" + type + "\\" + now.strftime("%Y_%m_%d--%H_%M_%S"), flush_secs=10)

        self.lossLog= {
            "train": [],
            "test": []
        }

        self.globalStep = {
            "train": 0,
            "test": 0
        }
        self.imStep = 0

        self.metrics = {}

    def addMetric(self, name, function):
        self.metrics[name] = function

    def computeMetrics(self, x, y, split):
        for metric in self.metrics.keys():
            temp = self.metrics[metric](x, y)
            self.writer.add_scalar(metric + "/" + split, temp, self.globalStep[split])

    def endOfBatch(self, x, y, loss, itter, split):
        self.writer.add_scalar("loss" + "/" + split, loss, self.globalStep[split])
        self.lossLog[split].append(loss.detach().cpu().numpy())
        self.computeMetrics(x, y, split)
        self.globalStep[split] += 1

    def endOfEpoch(self):
        for split in self.lossLog.keys():
            self.lossLog[split] = []
        self.imStep += 1

    def makeImage(self, item, y_hat, ii):
        names = ["viv1/", "viv2/", "sim1/" , "sim2/"]

        GT = IHATEPYTORCH(item["GT"])
        y_n = IHATEPYTORCH(item["y_n"])
        y_hat = IHATEPYTORCH(y_hat)["GT"]
        
        self.writer.add_image(names[ii] + "1. GT", y, self.imStep, dataformats='NCWH')
        self.writer.add_image(names[ii] + "2. Generated RealData", x, self.imStep, dataformats='NCWH')
        self.writer.add_image(names[ii] + "3. Generated AllData", x_art, self.imStep, dataformats='NCWH')

    def getProgbarDescription(self, split):
        tempstr = "loss: %.3e" % np.mean(self.lossLog[split])
        return tempstr

    def endOfTraining(self):
        self.writer.close()
        

def IHATEPYTORCH(x):
    min1 = np.amin(x, axis=(1, 2))
    x = x - min1.reshape(-1, 1, 1)

    max1 = np.amax(x, axis=(1, 2))
    x = x / max1.reshape(-1, 1, 1)

    x = np.expand_dims(x, 1)

    return x