from sbibm.tasks.two_moons.task import TwoMoons
import matplotlib.pyplot as plt
import pandas as pd
from sbibm.visualisation.metric import fig_metric_seaborn as fig_metric
from sbibm.visualisation.posterior import fig_posterior
from sbibm.algorithms.sbi.mcabc import run as rej_abc
from sbibm.metrics import c2st
from tqdm import tqdm
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

task = TwoMoons()
prior = task.get_prior()
simulator = task.get_simulator()
observation = task.get_observation(num_observation=10)  # 10 per task

data = []
for num_simulations in [1e3]:
    print(f"Running for {int(num_simulations)} simulations...")
    for num_observation in tqdm(range(1,2)):
        num_simulations = int(num_simulations)

        posterior_samples, _, _ = rej_abc(task=task, num_samples=10_000, num_observation=num_observation, num_simulations=num_simulations)

        """
        fig = fig_posterior(task=task,
                            num_observation=num_observation,
                            samples=[posterior_samples])
        plt.show()"""


        """
        sns.pairplot(pd.DataFrame(posterior_samples.numpy(), columns=["theta1", "theta2"]))
        plt.show()

        print(posterior_samples.shape)"""

        reference_samples = task.get_reference_posterior_samples(num_observation=num_observation)
        c2st_accuracy = c2st(reference_samples, posterior_samples).item()

        data.append({
            "algorithm": "REJ-ABC",
            "num_simulations": num_simulations,
            "C2ST": c2st_accuracy
        })

df = pd.DataFrame(data)

fig = fig_metric(
    df=df,
    metric="C2ST",
    title="C2ST of REJ-ABC on TwoMoons"
)
plt.show()