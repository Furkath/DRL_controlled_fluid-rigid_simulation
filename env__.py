import taichi as ti
import numpy as np
import math
import random
from shoot import MPMSolver

import jittor
from Autoencoder import AutoEncoder, normalize, inv_normalize
import torch
import os

ti.init(arch=ti.gpu)

class JellyEnv(MPMSolver):
    def __init__(self,n_tasks,path,gui):
        self.gui=gui
        self.n_tasks=n_tasks
        self.cnt=0
        self.model = AutoEncoder()
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        scale=0.8
        self.qualitys=np.random.randint(20,31,self.n_tasks)*0.1 #for meta train
        self.Es=np.random.randint(3,15,self.n_tasks)*100.0
        self.quality=float(self.qualitys[0])
        self.E=self.Es[0]
        
        srand=np.random.randint(1,4,size=(self.n_tasks,12))*0.4+1.0
        self.projections=np.random.randint(1,2,size=self.n_tasks)
        self.projection=self.projections[0]
        self.jelly_phos=srand*1.0
        self.jelly_pho=self.jelly_phos[0]
        
        srand=np.random.randint(8,9,self.n_tasks)*40.0
        self.fluid_phos=srand*0.01
        self.fluid_pho=self.fluid_phos[0]
        
        srand=np.zeros([self.n_tasks,3])
        for i in range(self.n_tasks):
            srand[i][0]=scale*(np.random.randint(0,3)*0.04+0.08)#length
            srand[i][1]=scale*(np.random.randint(0,1)*0.06+0.00)#height
            srand[i][2]=scale*(np.random.randint(0,3)*0.04+0.12)#bottom

        self.shapes=srand
        self.shape=self.shapes[0]
        
        srand=np.random.randint(1,4,self.n_tasks)*20.0#-20.0
        self.gravitys=srand
        self.gravity=self.gravitys[0]
        
        srand=np.zeros([self.n_tasks,10])
        for i in range(self.n_tasks):
            srand[i]=random.sample(range(0, 10),self.n_tasks)
        self.r_types=srand
        self.r_type=self.r_types[0]
        
        super().__init__(self.quality,self.E,self.jelly_pho,self.fluid_pho,self.gravity,self.projection,self.r_type,self.shape)
        self.observation_space=self.get_observation_space()
        self.action_space=np.zeros(2)
        
    def get_all_task_idx(self):
        return range(self.n_tasks)
    
    def reset_task(self,idx):
        
        ti.reset()
        ti.init(arch=ti.gpu) # Try to run on GPU
        self.quality=float(self.qualitys[idx])
        self.E=self.Es[idx]
        self.jelly_pho=self.jelly_phos[idx]
        self.fluid_pho=self.fluid_phos[idx]
        self.gravity=self.gravitys[idx]
        self.projection=self.projections[idx]
        self.r_type=self.r_types[idx]
        self.shape=self.shapes[idx]
        self.reset(self.quality,self.E,self.jelly_pho,self.fluid_pho,self.gravity,self.projection,self.r_type,self.shape)
        
        self.reset_env()

    def get_observation_space(self):
        alpha=0.6
        state=self.get_state().reshape([-1,2,128,128])
        mean = np.load('data/field.npz')['mean']
        std = np.load('data/field.npz')['std']
        sta = torch.tensor(state)
        sta = normalize(sta, mean, std)
        encode = self.model.encode(sta)
        encode = encode.cpu().detach().numpy()
        a = np.append(self.get_ball_state(), self.get_rigid_state())
        obs = np.append(a, alpha*encode)

        return obs
        
        
    def step(self,action):
        for s in range(int(12.0e-3 / self.dt+0.1)):
            self.penshui()
            self.solve_tube(action[0] * 45, 0, action[1] * 900)
            self.substep()
            self.solve_ball()

        obs = self.get_observation_space()
        reward = self.get_reward()
        done = 0

        if self.get_ball()[1] < 0.15:
            done = 1

        return obs,reward,done,dict(reward=reward)
        
    def reset_env(self):
        self.o=random.sample(range(0, 5), 5)
        self.initialize()
        return self.get_observation_space()
    
    def render(self):
        color_jelly=np.array([0xff7fff,
                              0xff99ff,
                              0xffb2ff,
                              0xffd1ff,
                              0xfff4ff,0x84bf96,0x000000],dtype=np.uint32)
        self.gui.circles(self.x.to_numpy(), radius=1.5, color=0x84bf96)
        self.gui.circles(self.tube_x.to_numpy(), radius=1.5,color=0x845538)
        self.gui.circles(self.ball_x.to_numpy(), radius=1.5,color=0xffb2ff)
        self.gui.show()
