import taichi as ti
import numpy as np
import math
import random
from shoot import MPMSolver

import jittor
from Autoencoder import AutoEncoder, normalize, inv_normalize
import torch
import os
# from jittor.autograd import Variable

ti.init(arch=ti.gpu)

class JellyEnv(MPMSolver):
    def __init__(self,n_tasks,path,gui):
        self.frame_zzw = 0 
        self.gui=gui
        self.n_tasks=n_tasks
        self.cnt=0
        self.model = AutoEncoder()
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        scale=0.8
        self.summ=0
        self.epo=0
        self.qualitys= 2.5*np.ones(self.n_tasks) + 0.005*np.random.randint(-5,6,self.n_tasks)
                      #np.random.randint(20,31,self.n_tasks)*0.1 #for meta train
        self.Es= 850*np.ones(self.n_tasks) + 0.5*np.random.randint(-5,6,self.n_tasks)
                #np.random.randint(3,15,self.n_tasks)*100.0
        self.aas=np.random.randint(10,11,self.n_tasks)*0.1
        self.quality=float(self.qualitys[0])
        self.E=self.Es[0]

        ###srand=np.random.randint(1,4,size=(self.n_tasks,12))*0.4+1.0
        #self.projections= np.ones(self.n_tasks)
        self.projections= np.random.randint(1,2,size=self.n_tasks)
        self.projection=self.projections[0]
        self.jelly_phos = 50*np.ones((self.n_tasks,12))+0.05*np.random.randint(-5,6,size=(self.n_tasks,12))
                         #srand*1.0
        self.jelly_pho=self.jelly_phos[0]
        #self.jelly_pho=[50]
        #self.jelly_pho=np.array([23, 17,  3, 16,  8, 20, 10,  4, 22, 19,  4, 18])*0.1
        
        #srand=np.random.randint(8,9,self.n_tasks)*40.0
        self.fluid_phos= 3.4*np.ones(self.n_tasks)+0.005*np.random.randint(-5,6,self.n_tasks)
                        #srand*0.01
        self.fluid_pho=self.fluid_phos[0]
        
        srand=np.zeros([self.n_tasks,3])
        for i in range(self.n_tasks):
            srand[i][0]=0.0325+0.00005*np.random.randint(-5,6)   #scale*(np.random.randint(0,3)*0.04+0.08)#length
            srand[i][1]=0     +0.00005*np.random.randint(-5,6)   #scale*(np.random.randint(0,1)*0.06+0.00)#height
            srand[i][2]=0.06  +0.00005*np.random.randint(-5,6)   #scale*(np.random.randint(0,3)*0.04+0.12)#bottom

        self.shapes=srand
        self.shape=self.shapes[0]
        
        srand= 20*np.ones(self.n_tasks) + 0.05*np.random.randint(-5,6,self.n_tasks)
              #np.random.randint(1,4,self.n_tasks)*20.0#-20.0
        self.gravitys=srand
        self.gravity=self.gravitys[0]
        
        srand=np.zeros([self.n_tasks,10])
        for i in range(self.n_tasks):
            srand[i]=random.sample(range(0, 10), 10)
        #srand=np.random.randint(0,2,size=(self.n_tasks,12))
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
        alpha=0.1
        state=self.get_state().reshape([-1,2,128,128])

        # state[np.isnan(state)] = 0
        # mask=np.zeros(state.shape)
        # mask[:,:,4:124,4:124]=1
        # state[mask==0]=0
        
        # sta = jittor.Var(state)
        # sta = jittor.Var(sta.float())
        #print(jittor.std(sta))
        #print(jittor.std(sta))
        # sta=sta/(jittor.std(sta)+1e-7)
        mean = np.load('data/field.npz')['mean']
        std = np.load('data/field.npz')['std']
        sta = torch.tensor(state)
        sta = normalize(sta, mean, std)
        encode = self.model.encode(sta)
        encode = encode.cpu().detach().numpy()
        #decode=decode.cpu().detach().numpy()
        #print(np.mean(np.square(decode-state)))
        
        #print(encode)
        
        a = np.append(self.get_ball_state(), self.get_rigid_state())
        #print('xxxxxxxxxxxxxxxxxxx')
        #print(encode)
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
        self.cnt += 1
        
        if self.cnt==118:
            self.epo+=1
            if self.epo%5!=1:
                self.summ+=self.maxx[None]
            #print(self.maxx[None],self.summ,self.epo)  

        if self.get_ball()[1] < 0.15:
            done = 1

        #print(reward)
        #if self.bounded[None]==1:
        #    done=1
        return obs,reward,done,dict(reward=reward)
        
    def reset_env(self):
        self.o=random.sample(range(0, 5), 5)
        #print(self.maxx[None])
        self.initialize()
        self.cnt=0
        return self.get_observation_space()
    
    def render(self):
        
        #colors = np.array([0x008B8B, 0xFF6347, 0xEEEEF0], dtype=np.uint32)
        color_jelly=np.array([0xff7fff,
                              0xff99ff,
                              0xffb2ff,
                              0xffd1ff,
                              0xfff4ff,0x84bf96,0x000000],dtype=np.uint32)
        self.gui.circles(self.x.to_numpy(), radius=1.5, color=0x84bf96)
        self.gui.circles(self.tube_x.to_numpy(), radius=1.5,color=0x845538)
        self.gui.circles(self.ball_x.to_numpy(), radius=1.5,color=0xffb2ff)
        self.gui.show()
        ###imgname = f"{self.frame_zzw:0>{6}}"
        ###self.frame_zzw += 1
        #self.gui.lines(self.begin.reshape([-1,2]),self.end.reshape([-1,2]), radius=1,color=0xd71345)
        ###self.gui.show('pics/'+imgname+'.png')
