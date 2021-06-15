import torch
import torch.nn as nn

from tqdm import tqdm
import numpy as np

class Trainer(nn.Module):

    def __init__(self, dataset, options, dataLogger, device, doTest = False):
        self.device = device
        self.dataset = dataset
        self.options = options

        self.dataLogger = dataLogger
        self.doTest = doTest

    def Step(self, network, itter, split):
        indexes = getShuffletIndexes(self.dataset.getLength(split))
        indexes = tqdm(indexes)
        for ii in indexes:
            item = self.dataset.getItem(ii, split, itter)
            y_n = torch.tensor(item["y_n"], device=self.device).unsqueeze(0)
            GT = torch.tensor(item["GT"], device=self.device).unsqueeze(0)

            y_hat = network.proxStep(y_n, itter)

            loss = self.options["criterion"](y_hat, GT)

            if split == "train":
                loss.backward()
                self.options["optimizer"].step()
                self.options["optimizer"].zero_grad() 
            elif split == "val":
                dataLogger.saveImages(item, y_hat.detach().cpu().numpy(), ii)

            dataLogger.endOfBatch(loss)
            indexes.set_description(dataLogger.getProgbarDescription(split))

            if self.doTest == True:
                break;

    @torch.no_grad()
    def saveStep(self, network, itter):

        for split in ["train", "test", "val"]:
            print("saving " + split + " images")
            indexes = np.arange(self.dataset.getLength(split))
            indexes = tqdm(indexes)
            for ii in indexes:
                item = self.dataset.getItem(ii, split, itter)
                y_0 = torch.tensor(item["y_0"], device=self.device).unsqueeze(0)
                y_n = torch.tensor(item["y_n"], device=self.device).unsqueeze(0)

                y_hat = network.POCSStep(y_n, itter)
                y_hat = y_hat + y_0

                saveItem = {
                    "data": y_hat.cpu().numpy()
                } 

                self.dataset.setItem(saveItem, ii, split, itter)
            
                if self.doTest == True:
                    break;
    @torch.no_grad()
    def firstStep(self, network):
        for split in ["train", "test", "val"]:
            print("saving " + split + " images")
            indexes = np.arange(self.dataset.getLength(split))
            indexes = tqdm(indexes)
            for ii in indexes:
                item = self.dataset.getItem(ii, split, -1)
                x = torch.tensor(item["x"], device=self.device).unsqueeze(0)

                saveItem = {
                    "data": network.firstStep(x).cpu().numpy()
                } 

                self.dataset.setItem(saveItem, ii, split, -1)
                
                if self.doTest == True:
                    break;

def getShuffledIndexes(length):
    indexes = np.arange(length)
    np.random.shuffle(indexes)
    return indexes