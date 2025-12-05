import torch.nn as nn


class BaseNetworkSumarized(nn.Module):
    def __init__(self):
        super(BaseNetworkSumarized, self).__init__()

    def forward(self, x, y):
        summary = self.summarize(x)
        metric = self.metric(x, y)

        return summary, metric
    
    def summarize(self, x):
        raise NotImplementedError("Summarize method not implemented.")
    
    def metric(self,x,y):
        raise NotImplementedError("Metric method not implemented.")
