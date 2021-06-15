import torch
import h5py

class PACTDataset():

    def __init__(self, path, S_real, S_art):
        self.real_idx = S_real.getIdx()
        self.art_idx = S_art.getIdx()
        self.x = h5py.File(path, 'r+')

    def getLength(self, split):
        return len(self.x[split]["x"])

    def getItem(self, idx, split, itter):
        if itter == -1:
            item = {
                "x" : self.x[split]["x"][idx][self.real_idx]
            }
        elif itter == 0:
            item = {
                "y_0": self.x[split]["y_0"][idx],
                "y_n": self.x[split]["y_0"][idx],
                "GT": self.x[split]["y"][idx]
            }
        else:
            item = {
                "y_0": self.x[split]["y_0"][idx],
                "y_n": self.x[split]["y_n"][idx],
                "GT": self.x[split]["y"][idx]
            }
        return item

    def setItem(self, item, idx, split, itter):
        if itter == -1:
            self.x[split]["y_0"][idx] = item["y_0"]
        else:
            self.x[split]["y_n"][idx] = item["y_n"]
