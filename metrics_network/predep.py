from metrics_network.base import BaseNetworkSumarized
import torch

class PredepNetwork(BaseNetworkSumarized):
    def __init__(self, encoder_model, decoder_model, num_samples=1000):
        super(PredepNetwork, self).__init__()
        self.encoder = encoder_model
        self.decoder = decoder_model
        self.num_samples = num_samples

    def summarize(self, x):
        z = self.encoder(x)
        return z

    def metric(self, x, y):
        z = self.summarize(x)
        z = self.decoder(z)

        episilon = z - y
        episilon_1 = episilon[torch.randint(0, episilon.shape[0], (self.num_samples,))]
        episilon_2 = episilon[torch.randint(0, episilon.shape[0], (self.num_samples,))]

        boots = episilon_1 - episilon_2
        cov = boots.T @ boots / (boots.shape[0] - 1)

        d = boots.size(1)
        det_cov = torch.linalg.det(cov)

        # Gaussian density at 0
        p0 = 1.0 / torch.sqrt((2 * torch.pi)**d * det_cov)

        return p0