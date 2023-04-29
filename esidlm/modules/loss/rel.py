import torch.nn as nn

class RelativeErrorLoss(nn.modules.loss._Loss):

    def __init__(self, a, b):
        super().__init__()
        self.a = a
        self.b = b

    def forward(self, outputs, targets):
        loss = (outputs - targets) / (self.b + self.a * targets)
        loss = (loss ** 2).mean()
        return loss
