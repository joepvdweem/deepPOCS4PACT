import torch
import torch.nn as nn

from tqdm import tqdm
import numpy as np

class Trainer(nn.Module):
    def __init__(self, dataset, options, dataLogger, device, doTest = False):
        """
        initialization for the trainer module

        Parameters
        ----------
        dataset : the dataset that is used for training
        options : options used for training consisting of:
            "criterion": the loss/cost function
            "optimizer": the optimizer (e.g. ADAM)
        dataLogger : a datalogger object
        device : the device that the code runs on (e.g. "cuda:0" for gpu 0)
        doTest (optional) : debugging aid. If true, only runs every step once. 
        """
        
        self.device = device
        self.dataset = dataset
        self.options = options

        self.dataLogger = dataLogger
        self.doTest = doTest

    def Step(self, network, itter, split):
        """
        function used to train, test and validate the network.

        Parameters
        ----------
        network : the network that needs to be evaluatedd
        itter : the itteration in the network
        split : either "train", "test" or "val" where
            "train": trains the network
            "test": tests it on different data
            "val": runs a couple of images for visual inspection per epoch in Tensorboard
        """
        
        #shuffle the data for training and testing but not for validating
        if split == "val":
            indexes = np.arange(self.dataset.getLength(split))    
        else:
            indexes = getShuffledIndexes(self.dataset.getLength(split))
        
        #setup tqdm for visualization
        indexes = tqdm(indexes)
        for ii in indexes:
            #get data
            item = self.dataset.getItem(ii, split, itter)
            y_n = torch.tensor(item["y_n"], device=self.device, dtype=torch.float32).unsqueeze(0)
            GT = torch.tensor(item["GT"], device=self.device, dtype=torch.float32).unsqueeze(0)
            
            #Run Network Proximal step for training
            y_hat = network.proxStep(y_n, itter)

            #calculate loss
            loss = self.options["criterion"](y_hat, GT)

            if split == "train":
                #run optimizer
                loss.backward()
                self.options["optimizer"].step()
                self.options["optimizer"].zero_grad() 
            elif split == "val":
                #save images
                self.dataLogger.saveImages(item, y_hat.detach().cpu().numpy(), ii)

            #append datalogger data
            self.dataLogger.endOfBatch(y_n, y_hat, loss, itter, split)
            indexes.set_description(self.dataLogger.getProgbarDescription(split))

            #debugging
            if self.doTest == True:
                break;

    @torch.no_grad()
    def saveStep(self, network, itter):
        """
        Saves all the data for the itterations by running the entire POCS-step.

        Parameters
        ----------
        network : The network that needs to be trained
        itter : The specific itteration that needs to be trained
        """
        
        for split in ["train", "test", "val"]:
            print("saving " + split + " images")
            indexes = np.arange(self.dataset.getLength(split))
            indexes = tqdm(indexes)
            for ii in indexes:
                #getitem
                item = self.dataset.getItem(ii, split, itter)
                y_0 = torch.tensor(item["y_0"], device=self.device, dtype=torch.float32).unsqueeze(0)
                y_n = torch.tensor(item["y_n"], device=self.device, dtype=torch.float32).unsqueeze(0)

                #run it through an entire pocs step (so network + forward/backward pass)
                y_hat = network.POCSStep(y_n, itter)
                
                #data consistency step
                y_hat = y_hat + y_0
                
                #save item in dataset
                saveItem = {
                    "y_n": y_hat.cpu().numpy()
                } 
                self.dataset.setItem(saveItem, ii, split, itter)
            
                if self.doTest == True:
                    break;
    @torch.no_grad()
    def firstStep(self, network):
        """
        The first step is transforming the acquired sensor data to image data by e.g. beamforming.

        Parameters
        ----------
        network : The network that contains the desired backward pass.
        """
        
        #for all the data
        for split in ["train", "test", "val"]:
            print("saving " + split + " images")
            indexes = np.arange(self.dataset.getLength(split))
            indexes = tqdm(indexes)
            for ii in indexes:
                #getitem
                item = self.dataset.getItem(ii, split, -1)
                x = torch.tensor(item["x"], device=self.device, dtype=torch.float32).unsqueeze(0)
                
                #run first step
                y_0 = network.firstStep(x)
                
                #save item
                saveItem = {
                    "y_0": y_0.cpu().numpy()
                } 
                self.dataset.setItem(saveItem, ii, split, -1)
                
                if self.doTest == True:
                    break;

def getShuffledIndexes(length):
    indexes = np.arange(length)
    np.random.shuffle(indexes)
    return indexes