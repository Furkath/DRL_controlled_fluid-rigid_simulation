import os
import pathlib
import numpy as np
import click
import json
import jittor
import taichi as ti
import random

from shoot import MPMSolver
from env import JellyEnv
from Autoencoder import AutoEncoder
from rlkit.jittor.sac.policies import TanhGaussianPolicy
from rlkit.jittor.networks import FlattenMlp, MlpEncoder
from rlkit.jittor.sac.sac import PEARLSoftActorCritic
from rlkit.jittor.sac.agent import PEARLAgent
from rlkit.launchers.launcher_util import setup_logger
import rlkit.jittor.jittor_util as ptu
from configs.default import default_config
from rlkit.jittor.sac.policies import MakeDeterministic
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as Data 
import torch.optim as optim
from torch.utils.data import Dataset

gui=ti.GUI("MPM SCOOP THE JELLY", res=(512,512), background_color=0x112F41, show_gui=True)

def setup_seed(seed=1024):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    jittor.set_global_seed(seed) 

setup_seed() 

if jittor.has_cuda:
    jittor.flags.use_cuda = 1

def experiment(variant):
    env = JellyEnv(1, 'model/autoencoder_full.pth', gui)
    deterministic = False
    tasks = env.get_all_task_idx()
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    reward_dim = 1

    latent_dim = variant['latent_size']
    context_encoder_input_dim = 2 * obs_dim + action_dim + reward_dim
    context_encoder_output_dim = latent_dim * 2 # information bottleneck
    net_size = variant['net_size']
    encoder_model = MlpEncoder

    context_encoder = encoder_model(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=context_encoder_input_dim,
        output_size=context_encoder_output_dim,
    )
    qf1 = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    )
    qf2 = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    )
    vf = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + latent_dim,
        output_size=1,
    )
    policy = TanhGaussianPolicy(
        hidden_sizes=[net_size, net_size, net_size],
        obs_dim=obs_dim + latent_dim,
        latent_dim=latent_dim,
        action_dim=action_dim,
    )
    agent = PEARLAgent(
        latent_dim,
        context_encoder,
        policy,
        **variant['algo_params']
    )

    algorithm = PEARLSoftActorCritic(
        env=env,
        train_tasks=list(tasks[:variant['n_train_tasks']]),
        eval_tasks=list(tasks[-variant['n_eval_tasks']:]),
        nets=[agent, qf1, qf2, vf],
        latent_dim=latent_dim,
        **variant['algo_params']
    )

    algorithm.train()
    qf1.save('model/qf1.pkl')
    qf2.save('model/qf2.pkl')
    vf.save('model/vf.pkl')
    policy.save('model/policy.pkl')
    context_encoder.save('model/context_encoder.pkl')

def deep_update_dict(fr, to):
    for k, v in fr.items():
        if type(v) is dict:
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to

@click.command()
@click.argument('config', default=None)
def main(config):
    variant = default_config
    if config:
        with open(os.path.join(config)) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)

    experiment(variant)

if __name__ == "__main__":
    main()
