import torch
import torch.nn as nn

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

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


class PredepNetwork(BaseNetworkSumarized):
    def __init__(self, encoder_model, decoder_model, num_samples=1000, bandwidth=0.1):
        super().__init__()
        self.encoder = encoder_model
        self.decoder = decoder_model
        self.num_samples = num_samples
        self.bandwidth = bandwidth  # KDE bandwidth h

    def summarize(self, x):
        return self.encoder(x)

    def metric(self, x, y):
        z = self.decoder(self.summarize(x))
        epsilon = z - y

        # Bootstrap sampling
        idx1 = torch.randint(0, epsilon.size(0), (self.num_samples,))
        idx2 = torch.randint(0, epsilon.size(0), (self.num_samples,))
        boots = epsilon[idx1] - epsilon[idx2]     # shape: (N, d)

        # Bandwidth h
        h = self.bandwidth

        # Compute ||boots/h||^2 for each sample
        scaled = boots / h
        sq_norm = torch.sum(scaled**2, dim=1)     # shape (N,)

        # Gaussian kernel K(u) = exp(-||u||^2 / 2)
        kernel_vals = torch.exp(-0.5 * sq_norm)

        # Density estimate at 0: (1/(N h^d)) * sum K(boots/h)
        d = boots.size(1)
        density = kernel_vals.mean() / (h**d)

        return density