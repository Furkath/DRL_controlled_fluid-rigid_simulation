import numpy as np
import copy
from rlkit.samplers.util import rollout
from rlkit.jittor.sac.policies import MakeDeterministic


class InPlacePathSampler(object):
    """
    A sampler that does not serialization for sampling. Instead, it just uses
    the current policy and environment as-is.

    WARNING: This will affect the environment! So
    ```
    sampler = InPlacePathSampler(env, ...)
    sampler.obtain_samples  # this has side-effects: env will change!
    ```
    """
    def __init__(self, env, policy, max_path_length):
        self.env = env
        self.policy = policy

        self.max_path_length = max_path_length

    def start_worker(self):
        pass

    def shutdown_worker(self):
        pass
    
    def obtain_samples(self, deterministic=False, max_samples=np.inf, max_trajs=np.inf, accum_context=True, resample=1):
        """
        Obtains samples in the environment until either we reach either max_samples transitions or
        num_traj trajectories.
        The resample argument specifies how often (in trajectories) the agent will resample it's context.
        """
        assert max_samples < np.inf or max_trajs < np.inf, "either max_samples or max_trajs must be finite"
        policy = MakeDeterministic(self.policy) if deterministic else self.policy
        paths = []
        n_steps_total = 0
        n_trajs = 0
        rewards=0
        while n_steps_total < max_samples and n_trajs < max_trajs:
            path = rollout(
                self.env, policy, max_path_length=self.max_path_length, accum_context=accum_context)
            # save the latent context that generated this trajectory
            path['context'] = policy.z.detach().cpu().numpy()
            paths.append(path)
            length=len(path['observations'])
            
            
            n_steps_total += length
            n_trajs += 1
            rewards+=np.sum(path['rewards'])
            # don't we also want the option to resample z ever transition?
            if n_trajs % resample == 0:
                policy.sample_z()
            
        return paths, n_steps_total,rewards

