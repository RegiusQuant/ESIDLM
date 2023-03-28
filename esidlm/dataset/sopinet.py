import torch
from torch.utils.data import Dataset


class SOPiNetDataset(Dataset):

    def __init__(self, x_cont, x_cate, x_time, y=None, mask=None):
        self.x_cont = torch.FloatTensor(x_cont)
        self.x_cate = torch.LongTensor(x_cate)
        self.x_time = torch.FloatTensor(x_time)

        if y is not None:
            self.y = torch.FloatTensor(y)
            self.mask = torch.FloatTensor(mask)
        else:
            self.y = None
            self.mask = None
 
    def __len__(self):
        return len(self.x_cont)

    def __getitem__(self, idx):
        item = {
            "X_CONT": self.x_cont[idx],
            "X_CATE": self.x_cate[idx],
            "X_TIME": self.x_time[idx]
        }
        if self.y is not None:
            item["Y"] = self.y[idx]
            item["MASK"] = self.mask[idx]
        return item
