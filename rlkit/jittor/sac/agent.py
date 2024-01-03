import numpy as np

import jittor
import jittor as jt
from jittor import nn as nn
from jittor import distributions
# import jittor.nn.functional as F

import rlkit.jittor.jittor_util as ptu


def _product_of_gaussians(mus, sigmas_squared):
    '''
    compute mu, sigma of product of gaussians
    '''
    sigmas_squared = jittor.clamp(sigmas_squared, min_v=1e-7)
    sigma_squared = 1. / jittor.sum(np.reciprocal(sigmas_squared), dim=0)
    mu = sigma_squared * jittor.sum(mus / sigmas_squared, dim=0)
    return mu, sigma_squared


def _mean_of_gaussians(mus, sigmas_squared):
    '''
    compute mu, sigma of mean of gaussians
    '''
    mu = jittor.mean(mus, dim=0)
    sigma_squared = jittor.mean(sigmas_squared, dim=0)
    return mu, sigma_squared


def _natural_to_canonical(n1, n2):
    ''' convert from natural to canonical gaussian parameters '''
    mu = -0.5 * n1 / n2
    sigma_squared = -0.5 * 1 / n2
    return mu, sigma_squared


def _canonical_to_natural(mu, sigma_squared):
    ''' convert from canonical to natural gaussian parameters '''
    n1 = mu / sigma_squared
    n2 = -0.5 * 1 / sigma_squared
    return n1, n2

class Normal:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
    
    def sample(self, sample_shape=None):
        return jt.normal(jt.array(self.mu), jt.array(self.sigma),size=sample_shape)
    
    def rsample(self, sample_shape=None):
        return jt.array(self.mu)+jt.array(self.sigma)*jt.normal(jt.zeros_like(self.mu), jt.ones_like(self.sigma),size=sample_shape)

    def log_prob(self, x):
        var = self.sigma**2
        log_scale = jt.safe_log(self.sigma)
        return -((x-self.mu)**2) / (2*var) - log_scale-np.log(np.sqrt(2*np.pi))
    
    def entropy(self):
        return 0.5+0.5*np.log(2*np.pi)+jt.safe_log(self.sigma)

class PEARLAgent(nn.Module):

    def __init__(self,
                 latent_dim,
                 context_encoder,
                 policy,
                 **kwargs
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.context_encoder = context_encoder
        self.policy = policy

        self.recurrent = kwargs['recurrent']
        self.use_ib = kwargs['use_information_bottleneck']
        self.sparse_rewards = kwargs['sparse_rewards']
        self.use_next_obs_in_context = kwargs['use_next_obs_in_context']

        # initialize buffers for z dist and z
        # use buffers so latent context can be saved along with model weights

        # self.register_buffer('z', jittor.zeros(1, latent_dim))
        # self.register_buffer('z_means', jittor.zeros(1, latent_dim))
        # self.register_buffer('z_vars', jittor.zeros(1, latent_dim))

        self.z=jittor.zeros(1, latent_dim).stop_grad()
        self.z_means=jittor.zeros(1, latent_dim).stop_grad()
        self.z_vars=jittor.zeros(1, latent_dim).stop_grad()

        self.clear_z()

    def clear_z(self, num_tasks=1):
        '''
        reset q(z|c) to the prior
        sample a new z from the prior
        '''
        # reset distribution over z to the prior
        mu = ptu.zeros(num_tasks, self.latent_dim)
        if self.use_ib:
            var = ptu.ones(num_tasks, self.latent_dim)
        else:
            var = ptu.zeros(num_tasks, self.latent_dim)
        self.z_means = mu
        self.z_vars = var
        # sample a new z from the prior
        self.sample_z()
        # reset the context collected so far
        self.context = None
        # reset any hidden state in the encoder network (relevant for RNN)
        self.context_encoder.reset(num_tasks)

    def detach_z(self):
        ''' disable backprop through z '''
        self.z = self.z.detach()
        if self.recurrent:
            self.context_encoder.hidden = self.context_encoder.hidden.detach()

    def update_context(self, inputs):
        ''' append single transition to the current context '''
        o, a, r, no, d, info = inputs
        if self.sparse_rewards:
            r = info['sparse_reward']
        o = ptu.from_numpy(o[None, None, ...])
        a = ptu.from_numpy(a[None, None, ...])
        r = ptu.from_numpy(np.array([r])[None, None, ...])
        no = ptu.from_numpy(no[None, None, ...])

        if self.use_next_obs_in_context:
            data = jittor.concat([o, a, r, no], dim=2)
        else:
            data = jittor.concat([o, a, r], dim=2)
        if self.context is None:
            self.context = data
        else:
            self.context = jittor.concat([self.context, data], dim=1)

    def compute_kl_div(self):
        ''' compute KL( q(z|c) || r(z) ) '''
        prior = distributions.Normal(ptu.zeros(self.latent_dim), ptu.ones(self.latent_dim))
        # print(prior)
        posteriors = [distributions.Normal(mu, jittor.sqrt(var)) for mu, var in zip(jittor.unbind(self.z_means), jittor.unbind(self.z_vars))]
        # print(posteriors)
        kl_divs = [distributions.kl_divergence(post, prior) for post in posteriors]
        # print(kl_divs)
        kl_div_sum = jittor.sum(jittor.stack(kl_divs))
        return kl_div_sum

    def infer_posterior(self, context):
        ''' compute q(z|c) as a function of input context and sample new z from it'''
        params = self.context_encoder(context)
        params = params.view(context.size(0), -1, self.context_encoder.output_size)
        # with probabilistic z, predict mean and variance of q(z | c)
        if self.use_ib:
            mu = params[..., :self.latent_dim]
            sigma_squared = nn.softplus(params[..., self.latent_dim:])
            z_params = [_product_of_gaussians(m, s) for m, s in zip(jittor.unbind(mu), jittor.unbind(sigma_squared))]
            self.z_means = jittor.stack([p[0] for p in z_params])
            self.z_vars = jittor.stack([p[1] for p in z_params])
        # sum rather than product of gaussians structure
        else:
            self.z_means = jittor.mean(params, dim=1)
        self.sample_z()

    def sample_z(self):
        if self.use_ib:
            posteriors = [Normal(m, jittor.sqrt(s)) for m, s in zip(jittor.unbind(self.z_means), jittor.unbind(self.z_vars))]
            z = [d.rsample() for d in posteriors]
            self.z = jittor.stack(z)
        else:
            self.z = self.z_means
        

    def get_action(self, obs, deterministic=False):
        ''' sample action from the policy, conditioned on the task embedding '''
        z = self.z
        #print(z)
        # obs = ptu.from_numpy(obs[None])
        obs = jittor.Var(obs[None]).float()
        in_ = jittor.concat([obs, z], dim=1)
        return self.policy.get_action(in_, deterministic=deterministic)

    def set_num_steps_total(self, n):
        self.policy.set_num_steps_total(n)

    def execute(self, obs, context):
        ''' given context, get statistics under the current policy of a set of observations '''
        self.infer_posterior(context)
        self.sample_z()

        task_z = self.z

        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        task_z = [z.repeat(b, 1) for z in task_z]
        task_z = jittor.concat(task_z, dim=0)

        # run policy, get log probs and new actions
        in_ = jittor.concat([obs, task_z.detach()], dim=1)
        policy_outputs = self.policy(in_, reparameterize=True, return_log_prob=True)

        return policy_outputs, task_z

    def log_diagnostics(self, eval_statistics):
        '''
        adds logging data about encodings to eval_statistics
        '''
        z_mean = np.mean(np.abs(ptu.get_numpy(self.z_means[0])))
        z_sig = np.mean(ptu.get_numpy(self.z_vars[0]))
        eval_statistics['Z mean eval'] = z_mean
        eval_statistics['Z variance eval'] = z_sig

    @property
    def networks(self):
        return [self.context_encoder, self.policy]




