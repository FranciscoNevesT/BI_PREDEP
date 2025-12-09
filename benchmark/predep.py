from sbibm.tasks.two_moons.task import TwoMoons
import matplotlib.pyplot as plt
import pandas as pd
from sbibm.visualisation.metric import fig_metric_seaborn as fig_metric
from sbibm.visualisation.posterior import fig_posterior
from sbibm.metrics import c2st
from tqdm import tqdm
from sbibm.algorithms.amortized.predep import PredepNetwork
from sbibm.algorithms.amortized.amortized import run as amortized_nn
import torch
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

# Configs
batch_size = 1000
num_epochs = 10000
lr = 1e-4
inter_dim = 2

task = TwoMoons()

data = []
all_posterior_samples = {}
for num_simulations in [1e3,1e4,1e5]:
    print(f"Running for {int(num_simulations)} simulations...")
    for num_observation in range(1,11):
        num_simulations = int(num_simulations)
        
        reference_samples = task.get_reference_posterior_samples(num_observation=num_observation)

        posterior_samples, _, _ = amortized_nn(task=task, num_samples=10_000, 
                                               num_observation=num_observation, 
                                               num_simulations=num_simulations,
                                               PredepNetwork=PredepNetwork,
                                               batch_size=batch_size,
                                               num_epochs=num_epochs,
                                               lr=lr,
                                               inter_dim=inter_dim)
        
        all_posterior_samples[(num_simulations, num_observation)] = posterior_samples
        

# Save all posterior samples
torch.save(all_posterior_samples, "benchmark/posterior/twomoons.pt")