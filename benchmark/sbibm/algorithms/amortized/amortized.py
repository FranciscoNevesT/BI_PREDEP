from typing import Optional, Tuple

import torch
from sbi.inference import MCABC

from sbibm.tasks.task import Task
from sbibm.utils.io import save_tensor_to_csv
import torch.nn as nn
import tqdm

from nflows.flows import Flow
from nflows.distributions import StandardNormal
from nflows.transforms import CompositeTransform, MaskedAffineAutoregressiveTransform, ReversePermutation

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[16, 16, 16], activation=nn.ReLU):
        super(MLP, self).__init__()
        layers = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            layers.append(activation())
            prev_dim = hdim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    
def create_flow(data_dim, context_dim, num_layers=5, hidden_features=16):
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


def run(
    task: Task,
    num_samples: int,
    num_simulations: int,
    num_observation: Optional[int] = None,
    observation: Optional[torch.Tensor] = None,
    batch_size: int = 1000,
    show_progress_bars: bool = False,
    PredepNetwork: Optional[nn.Module] = None,
    num_epochs : int = 1,
    lr: float = 1e-4,
    inter_dim: int = None,
) -> Tuple[torch.Tensor, int, Optional[torch.Tensor]]:

    prior = task.get_prior()
    simulator = task.get_simulator(max_calls=num_simulations)

    priors = prior(num_samples=num_simulations)
    
    xs = simulator(priors)


    dataloader = torch.utils.data.DataLoader(
        dataset=torch.utils.data.TensorDataset(priors, xs),
        batch_size=batch_size,
        shuffle=True,
    )
    
    if inter_dim is None:
        inter_dim = output_dim

    input_dim = xs.shape[1]
    output_dim = priors.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    enconder = MLP(input_dim=input_dim, output_dim=inter_dim).to(device)
    decoder = MLP(input_dim=inter_dim, output_dim=output_dim).to(device)

    predep_network = PredepNetwork(encoder_model=enconder, decoder_model=decoder, 
                                   num_samples=num_simulations, bandwidth = 1)
    flow_model = create_flow(data_dim=output_dim, context_dim=inter_dim).to(device)

    optimizer = torch.optim.Adam(list(predep_network.parameters()) + list(flow_model.parameters()), lr=lr)

    pbar = tqdm.tqdm(range(num_epochs), desc="Training Predep Network")

    for _ in pbar:
        predep_loss = []

        for batch_idx, (theta_batch, x_batch) in enumerate(dataloader):

            theta_batch = theta_batch.to(device)
            x_batch = x_batch.to(device)

            summary, metric = predep_network(x_batch, theta_batch)

            loss = -metric.mean()

            predep_loss.append(-metric.mean().item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        predep_loss_val = sum(predep_loss) / len(predep_loss)

        # Update tqdm bar text
        pbar.set_postfix({
            "Predep": f"{predep_loss_val:.4f}",
        })

    pbar = tqdm.tqdm(range(num_epochs), desc="Training Flow Model")
    for _ in pbar:
        log_prob_loss = []

        for batch_idx, (theta_batch, x_batch) in enumerate(dataloader):

            theta_batch = theta_batch.to(device)
            x_batch = x_batch.to(device)

            summary = predep_network.summarize(x_batch).detach()
            log_prob = flow_model.log_prob(theta_batch, summary)

            loss = -log_prob.mean()

            log_prob_loss.append(-log_prob.mean().item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        log_prob_loss_val = sum(log_prob_loss) / len(log_prob_loss)

        # Update tqdm bar text
        pbar.set_postfix({
            "LogProb": f"{log_prob_loss_val:.4f}",
        })

    if observation is None:
        observation = task.get_observation(num_observation)

    x_obs_tilde = predep_network.summarize(observation.to(device)).detach()

    posterior_samples = flow_model.sample(num_samples, x_obs_tilde).detach().cpu()[0]

    return posterior_samples, simulator.num_simulations, None