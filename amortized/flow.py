from nflows.flows import Flow
from nflows.distributions import StandardNormal
from nflows.transforms import CompositeTransform, MaskedAffineAutoregressiveTransform, ReversePermutation
from nflows.nn.nets import ResidualNet
from metrics_network.base import BaseNetworkSumarized


def create_flow(data_dim, context_dim, num_layers=10, hidden_features=128):
    transforms = []
    for _ in range(num_layers):
        transforms.append(ReversePermutation(features=data_dim))
        transforms.append(MaskedAffineAutoregressiveTransform(
            features=data_dim,
            hidden_features=hidden_features,
            context_features=context_dim
        ))
    transform = CompositeTransform(transforms)
    base_distribution = StandardNormal([data_dim])
    flow = Flow(transform, base_distribution)
    return flow

class AmortizedFlowModel:
    def __init__(self, data_dim, context_dim, summarize :BaseNetworkSumarized
                 , num_layers=10, hidden_features=128, device='cpu'):
        self.device = device
        self.flow = create_flow(data_dim, context_dim, num_layers, hidden_features).to(self.device)
        self.summarize = summarize.to(self.device)
        self.summarize = self.summarize.eval()

    def log_prob(self, x, context):
        return self.flow.log_prob(x, context)