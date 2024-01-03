import taichi as ti
import numpy as np
import math
#import random
import time

@ti.data_oriented
class MPMSolver:
    def __init__(self,quality,E_,mass,rho,G,P,R,S):
        self.reset(quality,E_,mass,rho,G,P,R,S)
    def reset(self,quality,E_,mass,rho,G,P,R,S):
        #eject:
        self.penshuiID = ti.field(dtype=ti.i32, shape=())
        self.flux=20
        #self.penshuiID
        #self.num_X=ti.field(dtype=ti.i32, shape=())
        #self.copynum = 0
        #self.NUMeject = 20
        #self.id_eject = ti.field(dtype=ti.i32, shape=self.NUMeject)

        self.P=P
        self.rho=rho
        self.G=G
        self.E_=E_
        self.quality = quality # Use a larger value for higher-res simulations

        self.n_particles, self.n_grid = int((4000+4000+4000) * self.quality ** 2), int(64 * self.quality)
        self.dx, self.inv_dx = 1 / self.n_grid, float(self.n_grid)
        self.sample_size=128
        self.sample_dx,self.sample_inv_dx=1/self.sample_size,self.sample_size
        self.dt = 2.0e-4 / self.quality
        self.p_vol = (self.dx * 0.5)**2

        self.p_rho = ti.field(dtype=ti.f32, shape=self.n_particles) # material density
        self.p_mass = ti.field(dtype=ti.f32, shape=self.n_particles) # material mass
        self.x = ti.Vector.field(2, dtype=ti.f32, shape=self.n_particles) # position
        self.v = ti.Vector.field(2, dtype=ti.f32, shape=self.n_particles) # velocity
        self.C = ti.Matrix.field(2, 2, dtype=ti.f32, shape=self.n_particles) # affine velocity field (strain rate)
        self.F = ti.Matrix.field(2, 2, dtype=ti.f32, shape=self.n_particles) # deformation gradient (strain)
        self.material = ti.field(dtype=ti.i32, shape=self.n_particles) # material id
        self.Jp = ti.field(dtype=ti.f32, shape=self.n_particles) # plastic deformation

        self.sample_v=ti.Vector.field(2, dtype=ti.f32, shape=(self.sample_size, self.sample_size))
        self.sample_m=ti.field(dtype=ti.f32, shape=(self.sample_size, self.sample_size))
        #self.mask=ti.field(dtype=ti.f32,shape=(self.sample_size,self.sample_size))

        self.grid_v = ti.Vector.field(2, dtype=ti.f32, shape=(self.n_grid, self.n_grid)) # grid node momentum/velocity
        self.grid_m = ti.field(dtype=ti.f32, shape=(self.n_grid, self.n_grid)) # grid node mass
        #self.fluid_m=ti.field(dtype=ti.f32, shape=(self.n_grid, self.n_grid))
        #self.ball_m=ti.field(dtype=ti.f32, shape=(self.n_grid, self.n_grid))
        
        self.n_ball = 1#if it is one ball
        #self.a=ti.Vector.field(3,dtype=ti.f32, shape=())
        self.ball_pos=ti.field(dtype=ti.f32, shape=2*self.n_ball)
        self.ball_vel=ti.field(dtype=ti.f32, shape=2*self.n_ball)
        #initial ball mass for each task
        self.bbball_mass=ti.field(dtype=ti.f32,shape=self.n_ball)
        self.R=ti.field(dtype=ti.int32,shape=self.n_ball)
        self.color=ti.field(dtype=ti.int32,shape=self.n_particles)
        #self.order=ti.field(dtype=ti.i32,shape=5)
        self.o=np.zeros([5])
        self.S=ti.field(dtype=ti.f32,shape=4)
        self.maxx=ti.field(dtype=ti.i32,shape=())
        
        #Ball
        self.grid_x = ti.Vector.field(2, dtype=ti.f32, shape=(self.n_grid, self.n_grid))
        #for i,j in self.grid_x:
        #    self.grid_x[0]=(i+0.5)*self.dx
        #    self.grid_x[1]=(j+0.5)*self.dx
        #print(self.grid_x)
        self.bounce = 1
        self.shear = 0.5
        self.ball_R = 0.07
        self.ball_mass = 0.1
        self.ball_I = 0.5*self.ball_mass*self.ball_R*self.ball_R
        self.ball_center = ti.Vector.field(2,dtype=ti.f32, shape=())
        self.ball_angle = ti.field(dtype=ti.f32, shape=())
        self.ball_omega = ti.field(dtype=ti.f32, shape=())
        self.ball_vel = ti.Vector.field(2,dtype=ti.f32, shape=())
        self.ball_n_particles=int(800* quality ** 2)
        self.ball_x=ti.Vector.field(2, dtype=ti.f32, shape=self.ball_n_particles)
        # Tube
        self.center = ti.Vector.field(2,dtype=ti.f32, shape=())
        self.angle = ti.field(dtype=ti.f32, shape=())
        self.omega = ti.field(dtype=ti.f32, shape=())      
        self.vel = ti.Vector.field(2,dtype=ti.f32, shape=())
        self.grid_assit = ti.Vector.field(2, dtype=ti.f32, shape=(self.n_grid, self.n_grid)) # grid node momentum/velocity
        #add
        self.l_down=ti.Vector.field(2,dtype=ti.f32, shape=())
        self.l_up=ti.Vector.field(2,dtype=ti.f32, shape=())
        self.r_down=ti.Vector.field(2,dtype=ti.f32, shape=())
        self.r_up=ti.Vector.field(2,dtype=ti.f32, shape=())

        self.outl_down=ti.Vector.field(2,dtype=ti.f32, shape=())
        self.outl_up=ti.Vector.field(2,dtype=ti.f32, shape=())
        self.outr_down=ti.Vector.field(2,dtype=ti.f32, shape=())
        self.outr_up=ti.Vector.field(2,dtype=ti.f32, shape=())

        self.tube_n_particles=int(480* quality ** 2)
        self.tube_x=ti.Vector.field(2, dtype=ti.f32, shape=self.tube_n_particles) # position
        self.tube_mass = ti.field(dtype=ti.f32, shape=self.tube_n_particles)
        self.radius = S[0]
        #add
        self.thick = 0.015*0.8
        self.depth = ti.max(self.thick, ti.min(0.12*0.8-self.thick, S[2]) )
        self.bottom = 0.12*0.8-self.depth
        self.ratio = self.thick*self.depth /( self.thick*self.depth + (self.thick+self.radius)*self.bottom )
        self.rigid_size1=int(self.tube_n_particles*0.5*self.ratio)
        self.rigid_size2=int(self.rigid_size1+self.tube_n_particles*0.5*self.ratio)
        self.rigid_size3=int(self.rigid_size2+self.tube_n_particles*0.5*(1.0-self.ratio))
        self.rigid_size4=int(self.rigid_size3+self.tube_n_particles*0.5*(1.0-self.ratio)*self.quality**2)

        self.fluid_size = int((4000+4000+4000)* self.quality ** 2)
        self.ball_size = int((self.n_particles-self.fluid_size)/self.n_ball) #n ball 
        self.point=ti.Vector.field(2,dtype=ti.f32,shape=4)

        for i in range(self.n_ball):
            self.bbball_mass[i]=mass[i]
            self.R[i]=0
    
    @ti.kernel
    def solve_ball(self):
        #gravity
        self.ball_vel[None]+=self.G*ti.Vector([0,-1])*self.dt*1.
        self.ball_vel[None][0]=ti.min(self.ball_vel[None][0],3.0)
        self.ball_vel[None][0]=ti.max(self.ball_vel[None][0],-3.0)
        self.ball_vel[None][1]=ti.min(self.ball_vel[None][1],3.0)
        self.ball_vel[None][1]=ti.max(self.ball_vel[None][1],-3.0)
        self.ball_omega[None]=ti.min(self.ball_omega[None],6.0)
        self.ball_omega[None]=ti.max(self.ball_omega[None],-6.0)

        self.ball_angle[None]+=self.dt*self.ball_omega[None]
        self.ball_center[None]+=self.dt*self.ball_vel[None]
   
        ball_top = self.ball_center[None][1]+self.ball_R
        ball_bottom = self.ball_center[None][1]-self.ball_R
        ball_left = self.ball_center[None][0]-self.ball_R
        ball_right = self.ball_center[None][0]+self.ball_R
    
        out1=0
        out2=0
        out3=0
        out4=0
        if ball_top > 0.98:
            out1=1
        if ball_bottom < 0.01:
            out2=1
        if ball_left < 0.02:
            out3=1
        if ball_right > 0.98:
            out4=1

        if out1>0:
            self.ball_vel[None][1]=-0.5*abs(self.ball_vel[None][1])
            self.ball_omega[None]=-0.*self.ball_omega[None]
        elif out2>0:
            self.ball_vel[None][1]=0.5*abs(self.ball_vel[None][1])
            self.ball_omega[None]=-0.*self.ball_omega[None]
        if out3>0:
            self.ball_vel[None][0]=ti.max(0.02, 0.5*abs(self.ball_vel[None][0]))
            self.ball_omega[None]=-0.*self.ball_omega[None]
        elif out4>0:
            self.ball_vel[None][0]=-ti.max(0.02, 0.5*abs(self.ball_vel[None][0]))
            self.ball_omega[None]=-0.*self.ball_omega[None]

        for pp in self.ball_x:
            wr = ti.Vector([self.ball_center[None][1]-self.ball_x[pp][1], self.ball_x[pp][0]-self.ball_center[None][0]])
            wr = wr*self.ball_omega[None]
            self.ball_x[pp]+=(self.ball_vel[None]+wr)*self.dt


    @ti.kernel
    def solve_tube(self,a:ti.f32,b:ti.f32,c:ti.f32):#action:[ax,ay,aw] initially, we let ay can just be 0 
        self.vel[None][0]+=self.dt*a
        self.vel[None][1]+=self.dt*b
        self.omega[None]+=self.dt*c 

        self.vel[None][0]=ti.min(self.vel[None][0],3.0)
        self.vel[None][0]=ti.max(self.vel[None][0],-3.0)
        self.vel[None][1]=ti.min(self.vel[None][1],3.0)
        self.vel[None][1]=ti.max(self.vel[None][1],-3.0)
        self.omega[None]=ti.min(self.omega[None],6.0)
        self.omega[None]=ti.max(self.omega[None],-6.0)

        self.angle[None]+=self.dt*self.omega[None]
        self.center[None]+=self.dt*self.vel[None]
         
        #rotation matrix, not used imediately
        #rot1 = ti.Vector([ti.cos(self.angle[None]), -ti.sin(self.angle[None])])
        #rot2 = ti.Vector([ti.sin(self.angle[None]), ti.cos(self.angle[None])])
         
        wr = ti.Vector([self.center[None][1]-self.l_up[None][1], self.l_up[None][0]-self.center[None][0]])
        wr = wr*self.omega[None]
        self.l_up[None] += (self.vel[None]+wr)*self.dt
        wr = ti.Vector([self.center[None][1]-self.l_down[None][1], self.l_down[None][0]-self.center[None][0]])
        wr = wr*self.omega[None]
        self.l_down[None] += (self.vel[None]+wr)*self.dt
        wr = ti.Vector([self.center[None][1]-self.r_up[None][1], self.r_up[None][0]-self.center[None][0]])
        wr = wr*self.omega[None]
        self.r_up[None] += (self.vel[None]+wr)*self.dt
        wr = ti.Vector([self.center[None][1]-self.r_down[None][1], self.r_down[None][0]-self.center[None][0]])
        wr = wr*self.omega[None]
        self.r_down[None] += (self.vel[None]+wr)*self.dt
        
        wr = ti.Vector([self.center[None][1]-self.outl_up[None][1], self.outl_up[None][0]-self.center[None][0]])
        wr = wr*self.omega[None]
        self.outl_up[None] += (self.vel[None]+wr)*self.dt
        wr = ti.Vector([self.center[None][1]-self.outl_down[None][1], self.outl_down[None][0]-self.center[None][0]])
        wr = wr*self.omega[None]
        self.outl_down[None] += (self.vel[None]+wr)*self.dt
        wr = ti.Vector([self.center[None][1]-self.outr_up[None][1], self.outr_up[None][0]-self.center[None][0]])
        wr = wr*self.omega[None]
        self.outr_up[None] += (self.vel[None]+wr)*self.dt
        wr = ti.Vector([self.center[None][1]-self.outr_down[None][1], self.outr_down[None][0]-self.center[None][0]])
        wr = wr*self.omega[None]
        self.outr_down[None] += (self.vel[None]+wr)*self.dt

        out1=0
        out2=0
        out3=0
        out4=0
        
        if self.l_up[None][0]<0.02:
            out1=1
        elif self.l_up[None][0]>0.98:
            out2=2
        elif self.l_up[None][1]<0.01:
            out3=3
        elif self.l_up[None][1]>0.96:
            out4=4
        elif self.l_down[None][0]<0.02:
            out1=1
        elif self.l_down[None][0]>0.98:
            out2=2
        elif self.l_down[None][1]<0.01:
            out3=3
        elif self.l_down[None][1]>0.96:
            out4=4
        elif self.r_up[None][0]<0.02:
            out1=1
        elif self.r_up[None][0]>0.98:
            out2=2
        elif self.r_up[None][1]<0.01:
            out3=3
        elif self.r_up[None][1]>0.96:
            out4=4
        elif self.r_down[None][0]<0.02:
            out1=1
        elif self.r_down[None][0]>0.98:
            out2=2
        elif self.r_down[None][1]<0.01:
            out3=3
        elif self.r_down[None][1]>0.96:
            out4=4
        if self.outl_up[None][0]<0.02:
            out1=1
        elif self.outl_up[None][0]>0.98:
            out2=2
        elif self.outl_up[None][1]<0.01:
            out3=3
        elif self.outl_up[None][1]>0.96:
            out4=4
        if self.outl_down[None][0]<0.02:
            out1=1
        elif self.outl_down[None][0]>0.98:
            out2=2
        if self.outl_down[None][1]<0.01:
            out3=3
        elif self.outl_down[None][1]>0.96:
            out4=4
        elif self.outr_up[None][0]<0.02:
            out1=1
        if self.outr_up[None][0]>0.98:
            out2=2
        elif self.outr_up[None][1]<0.01:
            out3=3
        elif self.outr_up[None][1]>0.96:
            out4=4
        elif self.outr_down[None][0]<0.02:
            out1=1
        if self.outr_down[None][0]>0.98:
            out2=2
        if self.outr_down[None][1]<0.01:
            out3=3
        elif self.outr_down[None][1]>0.96:
            out4=4
        
        if out1>0:
            self.vel[None][0]=0.5*abs(self.vel[None][0])
            self.omega[None]=ti.min(-0.1*self.omega[None], -0.1)
        elif out2>0:
            self.vel[None][0]=-0.5*abs(self.vel[None][0])
            self.omega[None]=ti.max(-0.1*self.omega[None], 0.1)
        if out3>0:
            self.vel[None][1]=0.5*abs(self.vel[None][1])
            if self.outr_down[None][1] < self.center[None][1]:
                self.omega[None]= 0.1
            else:
                self.omega[None]=-0.1
        elif out4>0:
            self.vel[None][1]=-0.5*abs(self.vel[None][1])
            self.omega[None]=-0.1*self.omega[None]

        for p in self.tube_x:
            wr = ti.Vector([self.center[None][1]-self.tube_x[p][1], self.tube_x[p][0]-self.center[None][0]])
            wr = wr*self.omega[None]
            self.tube_x[p]+=(self.vel[None]+wr)*self.dt
        self.point[0]=self.outl_up[None]
        self.point[1] = self.outr_up[None]
        self.point[2] = self.outl_down[None]
        self.point[3] = self.outr_down[None]

    @ti.kernel
    def substep(self):
        #numeject = 0
        #print(self.NUMeject)
        #self.id_eject = ti.field(dtype=ti.i32, shape=self.NUMeject) 
        #for idd in self.id_eject:
        #    self.id_eject[idd] = -1
            #print(self.id_eject[idd])
        #ti.loop_config(serialize=True)
        #for p in range(0,self.n_particles):#self.x:
        #    if self.x[p][1]<0. and numeject < self.NUMeject:
                #print(numeject)
        #        self.id_eject[numeject]=p
        #        numeject +=1
                #print(numeject)
       # print(numeject)

        #for idd in self.id_eject:
        #    if self.id_eject[idd] > -1:
        #        pid = self.id_eject[idd]
        #        self.x[pid]= ti.random()*( self.r_down[None] - self.l_down[None] ) + self.l_down[None]
        #        self.v[pid]= ( self.l_up[None] - self.l_down[None] )*0.1

        E, nu = self.E_, 0.2 # Young's modulus and Poisson's ratio
        mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1+nu) * (1 - 2 * nu)) # Lame parameters
        for i, j in self.grid_m:
            if i<self.sample_size and j<self.sample_size:
                self.sample_v[i,j]=[0,0]
                self.sample_m[i,j]=0
            self.grid_v[i, j] = [0, 0]
            self.grid_m[i, j] = 0
            self.grid_assit[i, j] = [0, 0]
        ti.loop_config(block_dim=32)#32
        for p in self.x: # Particle state update and scatter to grid (P2G)
            #ppxx
            if self.x[p][1] > 0. and self.x[p][0] > 1.5*self.dx and self.x[p][0] < 1-1.5*self.dx:
                base = (self.x[p] * self.inv_dx - 0.5).cast(int)
                fx = self.x[p] * self.inv_dx - base.cast(float)
                # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
                w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
                base_sample = (self.x[p] * self.sample_inv_dx - 0.5).cast(int)
                fx_sample = self.x[p] * self.sample_inv_dx - base_sample.cast(float)
                # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
                sample_w = [0.5 * (1.5 - fx_sample) ** 2, 0.75 - (fx_sample - 1) ** 2, 0.5 * (fx_sample - 0.5) ** 2]

                self.F[p] = (ti.Matrix.identity(ti.f32, 2) + self.dt * self.C[p]) @ self.F[p] # deformation gradient update
                h = ti.exp(10 * (1.0 - self.Jp[p])) # Hardening coefficient: snow gets harder when compressed
                if self.material[p] == 1: # jelly, make it softer
                    h = 1.0
                mu, la = mu_0 * h, lambda_0 * h
                if self.material[p] == 0: # liquid
                    mu = 0.0
                U, sig, V = ti.svd(self.F[p])
                J = 1.0
                for d in ti.static(range(2)):
                    #new_sig = sig[d, d]
                    #self.Jp[p] *= sig[d, d] / new_sig
                    #sig[d, d] = new_sig
                    #J *= new_sig
                    J*=sig[d, d]
                if self.material[p] == 0:  # Reset deformation gradient to avoid numerical instability
                    self.F[p] = ti.Matrix.identity(ti.f32, 2) * ti.sqrt(J)
                elif self.material[p] == 2:
                    self.F[p] = U @ sig @ V.transpose() # Reconstruct elastic deformation gradient after plasticity
                stress = 2 * mu * (self.F[p] - U @ V.transpose()) @ self.F[p].transpose() + ti.Matrix.identity(ti.f32, 2) * la * J * (J - 1)
                stress = (-self.dt * self.p_vol * 4 * self.inv_dx * self.inv_dx) * stress
                affine = stress + self.p_mass[p] * self.C[p]

                if self.material[p] == 0 and self.P:
                    self.Jp[p]=(1 + self.dt * self.C[p].trace()) * self.Jp[p]
                    st = -self.dt * 4 * E * self.p_vol * (self.Jp[p] - 1) / self.dx**2
                    affine = ti.Matrix([[st, 0], [0, st]]) + self.p_mass[p] * self.C[p]

                for i, j in ti.static(ti.ndrange(3, 3)): # Loop over 3x3 grid node neighborhood
                    #if (base_sample + offset)[0] > -1 and (base_sample + offset)[0] < self.sample_size and (base_sample + offset)[1] > -1 and (base_sample + offset)[1] < self.sample_size:
                    offset = ti.Vector([i, j])
                    if (base_sample + offset)[0] > -1 and (base_sample + offset)[0] < self.sample_size and (base_sample + offset)[1] > -1 and (base_sample + offset)[1] < self.sample_size: 
                        #dpos = (offset.cast(float) - fx) * self.dx
                        #weight = w[i][0] * w[j][1]

                        dpos_sample=(offset.cast(float) - fx_sample) * self.sample_dx
                        weight_sample = sample_w[i][0] * sample_w[j][1]
                    #if (base_sample + offset)[0] > -1 and (base_sample + offset)[0] < self.sample_size and (base_sample + offset)[1] > -1 and (base_sample + offset)[1] < self.sample_size:
                        self.sample_v[base_sample + offset] += weight_sample * (self.p_mass[p] * self.v[p] + affine @ dpos_sample)
                        if self.material[p] == 1:
                            self.sample_v[base_sample + offset][1] -= weight_sample * self.dt * self.p_mass[p] * self.G * 2.50
                        else:
                            self.sample_v[base_sample + offset][1] -= weight_sample * self.dt * self.p_mass[p] * self.G
                        self.sample_m[base_sample + offset] += weight_sample * self.p_mass[p]
                    if (base + offset)[0] > -1 and (base + offset)[0] < self.n_grid and (base + offset)[1] > -1 and (base + offset)[1] < self.n_grid:
                        dpos = (offset.cast(float) - fx) * self.dx
                        weight = w[i][0] * w[j][1]
                        self.grid_assit[base + offset] += weight * (affine @ dpos)
                        self.grid_v[base + offset] += weight * (self.p_mass[p] * self.v[p] + affine @ dpos)
                        if self.material[p] == 1:
                            self.grid_v[base + offset][1] -= weight * self.dt * self.p_mass[p] * self.G * 2.50
                        else:
                            self.grid_v[base + offset][1] -= weight * self.dt * self.p_mass[p] * self.G
                        self.grid_m[base + offset] += weight * self.p_mass[p]

        ti.loop_config(block_dim=32)#32
        for p in self.tube_x:
            tube_C = ti.Matrix.zero(ti.f32, 2, 2)
            tube_F = ti.Matrix.identity(ti.f32, 2)  #ti.Matrix([1., 0.],[0., 1.])
            base = (self.tube_x[p] * self.inv_dx - 0.5).cast(int)
            fx = self.tube_x[p] * self.inv_dx - base.cast(float)
            # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
            h = 1.0 # Hardening coefficient: snow gets harder when compressed
            mu, la = mu_0 * h, lambda_0 * h
            U, sig, V = ti.svd(tube_F)
            J = 1.0
            stress = 2 * mu * (tube_F - U @ V.transpose()) @ tube_F.transpose() + ti.Matrix.identity(ti.f32, 2) * la * J * (J - 1)
            stress = (-self.dt * self.p_vol * 4 * self.inv_dx * self.inv_dx) * stress
            affine = stress + self.tube_mass[p] * tube_C
            wr = ti.Vector([self.center[None][1]-self.tube_x[p][1], self.tube_x[p][0]-self.center[None][0]])
            wr *= self.omega[None]
            new_vel=self.vel[None]+wr
            for i, j in ti.static(ti.ndrange(3, 3)): # Loop over 3x3 grid node neighborhood
                offset = ti.Vector([i, j])
                #dpos = (offset.cast(float) - fx) * self.dx
                #weight = w[i][0] * w[j][1]
                #wr = ti.Vector([self.center[None][1]-self.tube_x[p][1], self.tube_x[p][0]-self.center[None][0]])
                #wr = wr*self.omega[None]
                #new_vel=self.vel[None]+wr
                if (base + offset)[0] > -1 and (base + offset)[0] < self.n_grid and (base + offset)[1] > -1 and (base + offset)[1] < self.n_grid:
                    dpos = (offset.cast(float) - fx) * self.dx
                    weight = w[i][0] * w[j][1]
                    self.grid_v[base + offset] += weight * (self.tube_mass[p] * new_vel + affine @ dpos)
                    self.grid_m[base + offset] += weight * self.tube_mass[p]

        for i, j in self.grid_m:
            if self.grid_m[i, j] > 0: # No need for epsilon here
                self.grid_v[i, j] = (1 / self.grid_m[i, j]) * self.grid_v[i, j] # Momentum to velocity
                #zzw: self.grid_v[i, j][1] -= self.dt * self.G # gravity
            self.grid_v[i,j][1]=ti.min(self.grid_v[i,j][1],10)
            self.grid_v[i,j][0]=ti.min(self.grid_v[i,j][0],10)
            self.grid_v[i,j][1]=ti.max(self.grid_v[i,j][1],-10)
            self.grid_v[i,j][0]=ti.max(self.grid_v[i,j][0],-10)

            if i<self.sample_size and j<self.sample_size:
                if self.sample_m[i, j] > 1e-8: # No need for epsilon here
                    self.sample_v[i, j] = (1 / self.sample_m[i, j]) * self.sample_v[i, j] # Momentum to velocity
                    #zzw: self.sample_v[i, j][1] -= self.dt * self.G
                self.sample_v[i,j][1]=ti.min(self.sample_v[i,j][1],10)
                self.sample_v[i,j][0]=ti.min(self.sample_v[i,j][0],10)
                self.sample_v[i,j][1]=ti.max(self.sample_v[i,j][1],-10)
                self.sample_v[i,j][0]=ti.max(self.sample_v[i,j][0],-10)
            #BC:

            to_ball_distance=ti.math.distance(self.grid_x[i,j] , self.ball_center[None])
            if to_ball_distance <= self.ball_R:
                unit_dirc = (self.grid_x[i,j]-self.ball_center[None])/to_ball_distance
                bwr = ti.Vector([self.ball_center[None][1]-self.grid_x[i,j][1], self.grid_x[i,j][0]-self.ball_center[None][0]])
                bwr = bwr*self.ball_omega[None]
                ball_pointV=self.ball_vel[None]+bwr
                component = (self.grid_v[i,j] - ball_pointV).dot(unit_dirc)
                tanget_v = self.grid_v[i,j]-ball_pointV - component*unit_dirc
                faxiang = (1.+self.bounce)*unit_dirc*ti.min(0.0, component)
                qiexiang = (1.-self.shear)*tanget_v
                self.grid_v[i,j] -= faxiang
                self.grid_v[i,j] -= qiexiang
                self.ball_vel[None] += faxiang*self.grid_m[i,j]/self.ball_mass
                self.ball_omega[None] += ti.math.cross(unit_dirc, qiexiang)*to_ball_distance*self.grid_m[i,j]/self.ball_I
            
         
            #if i < self.quality and self.grid_v[i, j][0] < 0:
            #    n = ti.Vector([1, 0])
            #    nv = self.grid_v[i, j].dot(n)
            #    self.grid_v[i, j] = self.grid_v[i, j] - n * nv
            #    self.grid_v[i, j][0] =0.
                #self.grid_v[i, j] = ti.Vector([0, 0]) 
            #if i > self.n_grid - self.quality and self.grid_v[i, j][0] > 0:
            #    n = ti.Vector([-1, 0])
            #    nv = self.grid_v[i, j].dot(n)
            #    self.grid_v[i, j] = self.grid_v[i, j] - n * nv
            #    self.grid_v[i, j][0] =0.
                #self.grid_v[i, j] = ti.Vector([0, 0])
            #if j < self.quality and self.grid_v[i, j][1] < 0:
            #    n = ti.Vector([0, 1])
            #    nv = self.grid_v[i, j].dot(n)
            #    self.grid_v[i, j] = self.grid_v[i, j] - n * nv
                #self.grid_v[i, j] = ti.Vector([0, 0])
            if j > self.n_grid - self.quality  and self.grid_v[i, j][1] > 0:
                n = ti.Vector([0, -1])
                nv = self.grid_v[i, j].dot(n)
                self.grid_v[i, j] = self.grid_v[i, j] - n * nv
                #self.grid_v[i, j] = ti.Vector([0, 0])
    
        #self.num_X[None]=0
        ti.loop_config(block_dim=32) #32
        for p in self.x: # grid to particle (G2P)
            #ppxx
            if self.x[p][1] > 0. and self.x[p][0] > 1.5*self.dx and self.x[p][0] < 1-1.5*self.dx:
               # if self.x[p][0] > 0. and self.x[p][0] < 1.:
               #     self.num_X[None]+=1
               #     #print("x",self.x[p])
                base = (self.x[p] * self.inv_dx - 0.5).cast(int)
                fx = self.x[p] * self.inv_dx - base.cast(float)
                w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
                new_v = ti.Vector.zero(ti.f32, 2)
                new_C = ti.Matrix.zero(ti.f32, 2, 2)
                for i, j in ti.static(ti.ndrange(3, 3)): # loop over 3x3 grid node neighborhood
                    #dpos = ti.Vector([i, j]).cast(float) - fx
                    #g_v = ti.Vector([0., 0.]) 
                    if (base + ti.Vector([i, j]))[0] > -1 and (base + ti.Vector([i, j]))[0] < self.n_grid and (base + ti.Vector([i, j]))[1] > -1 and (base + ti.Vector([i, j]))[1] < self.n_grid:
                        dpos = ti.Vector([i, j]).cast(float) - fx
                        g_v = self.grid_v[base + ti.Vector([i, j])]
                        weight = w[i][0] * w[j][1]
                        new_v += weight * g_v
                        new_C += 4 * self.inv_dx * weight * g_v.outer_product(dpos)
                self.v[p], self.C[p] = new_v, new_C
                self.x[p] += self.dt * self.v[p] # advection
        #print(self.num_X[None])
        #self.copynum = self.num_X[None]
       
   #@ti.kernel
   # def getX(self):
   #     countX = 0
   #     gotx = ti.Vector.field(2, dtype=ti.f32, shape=self.num_X[None])
   #     #self.num_X[None]=0
   #     #print("OK")
   #     #ii = self.num_X[None]
   #     #print(ii)
   #     #intt = self.num_X.to_numpy()
   #     #print(intt)
   #     #print(self.num_X[None])

   #     #xpr=0.5*np.ones((int(self.num_X[None]), 2))
   #     #for xp in xtor:
   #         #print(xp)
   #         #if xp[1] > 0 and xp[0] > 0 and xp[0] < 1:
   #         #    np.vstack([xpr,xp])
   #     ti.loop_config(serialize=True)
   #     for p in range(0,self.n_particles): # grid to particle (G2P)
   #         if self.x[p][1] > 0. and self.x[p][0] > 0 and self.x[p][0] < 1: 
   #             gotx[countX] = self.x[p]
   #             #gotx[countX]=self.x[p]
   #             countX +=1
   #     return gotx
    @ti.kernel
    def penshui(self):
        for p in self.x:
            if ( p >= self.penshuiID[None]  and p < (self.penshuiID[None] + self.flux) ) or ( (p+self.n_particles) >= self.penshuiID[None]  and (p+self.n_particles) < (self.penshuiID[None] + self.flux)  ):
                self.x[p] = (self.r_down[None]-self.l_down[None])*ti.random()+self.l_down[None]
                self.v[p]=  (self.outr_up[None]-self.outr_down[None])*1000000
        self.penshuiID[None]+=self.flux
        if self.penshuiID[None] >= self.n_particles:
            self.penshuiID[None] -= self.n_particles

    @ti.kernel
    def initialize(self):
        for i,j in self.grid_x:
            self.grid_x[i,j][0]=(i+0.5)*self.dx
            self.grid_x[i,j][1]=(j+0.5)*self.dx
        #print(self.grid_x[0,0])
        #print(self.grid_x[0,self.n_grid-1])
        #print(self.grid_x[self.n_grid-1,0])
        #print(self.grid_x[self.n_grid-1,self.n_grid-1])
        self.maxx[None]=0
        for i, j in self.grid_m:
            if i<self.sample_size and j<self.sample_size:
                self.sample_v[i,j]=[0,0]
            self.grid_v[i, j] = [0, 0]
            self.grid_m[i, j] = 0
            self.grid_assit[i, j] = [0, 0]
        #eject:
        self.penshuiID[None] = 0
        '''
        a1=2-ti.random()*4
        a2=2-ti.random()*4
        for i in range(self.n_particles):
            if i<self.fluid_size:
                if i<0.5*self.fluid_size:
                    self.x[i] = [0.05+ti.random()*0.9  , 0.35+0.25*ti.random() ]
                    self.v[i]=ti.Matrix([a1,0])
                else:
                    self.x[i] = [0.05+ti.random()*0.9  , 0.66+0.25*ti.random() ]
                    self.v[i]=ti.Matrix([a2,0])
           # else:
           #     self.x[i] = [0. , -0.1 ]
           #     self.v[i]=ti.Matrix([0,0])
        '''
        for i in range(self.n_particles):
            self.x[i] = [0.  ,  -0.1]
            self.v[i]=ti.Matrix([0,0])
            self.p_rho[i]=self.rho
            self.p_mass[i]=self.p_rho[i]*self.p_vol
            self.material[i] = 0 # 0: fluid 1: jelly 
            self.color[i]=5
            self.F[i] = ti.Matrix([[1, 0], [0, 1]])
            self.Jp[i] = 1


        #ball
        p=0.95-0.55*ti.random()
        self.ball_center[None] = [p,0.95-self.ball_R]
        self.ball_angle[None] = 0.
        self.ball_omega[None] = 0#-2.5
        self.ball_vel[None]=ti.Vector([0.0,0.0])
        for i in range(self.ball_n_particles):
            #self.ball_mass[i]=self.p_vol*1000  ball_mass/n_ball
            rrr=ti.random()*self.ball_R
            ttt=ti.random()*2*math.pi
            self.ball_x[i]=[rrr*ti.cos(ttt) +self.ball_center[None][0], rrr*ti.sin(ttt) +self.ball_center[None][1] ]

        p=0.05+0.55*ti.random()
        '''
        print("Tube position is: ", p)
        print("Tube radius is: ", self.radius)
        print("Tube depth is: ", self.depth)
        print("Tube bottom is: ", self.bottom)
        print("Tube ratio is: ", self.ratio)
        print("Tube size1 is: ", self.rigid_size1)
        print("Tube size2 is: ", self.rigid_size2)
        print("Tube size3 is: ", self.rigid_size3)
        print("Tube Allsize is: ", self.tube_n_particles)
        '''
        self.center[None] = [p,0.03+self.S[1]]
        #may be for A3C:
        #center_point=self.center[None]
        self.l_down[None]=self.center[None] + ti.Vector([-self.radius,self.bottom])
        self.l_up[None]=self.l_down[None] + ti.Vector([0.0,self.depth])
        self.r_down[None]=self.center[None] + ti.Vector([self.radius,self.bottom])
        self.r_up[None]=self.r_down[None] + ti.Vector([0.0,self.depth]) 
        self.outl_down[None]=self.center[None] + ti.Vector([-self.radius,0.])#-self.thick,0.])
        self.outl_up[None]=self.outl_down[None] + ti.Vector([0,self.depth])#([-self.thick,self.depth])
        self.outr_down[None]=self.center[None] + ti.Vector([self.radius,0.])#+self.thick,0.])
        self.outr_up[None]=self.outr_down[None] + ti.Vector([0,self.depth])#([self.thick,self.depth]) 
        self.angle[None] = 0.
        self.omega[None] = 0#-2.5
        self.vel[None]=ti.Vector([3.0,0.0])
        for i in range(self.tube_n_particles):
            self.tube_mass[i]=self.p_vol*1000
            #self.tube_x[i]=[ti.random() * 2 * self.radius + self.outl_down[None][0] , ti.random() * self.depth + self.outl_down[None][1] ]
            if i<self.rigid_size1:
                self.tube_x[i]=[-ti.random() * self.thick + self.l_down[None][0] , ti.random() * self.depth + self.l_down[None][1] ]
            elif i<self.rigid_size2:
                self.tube_x[i]=[ti.random() * self.thick + self.r_down[None][0] , ti.random() * self.depth + self.r_down[None][1] ]
            elif i<self.rigid_size3:
                self.tube_x[i]=[-ti.random() * (self.thick + self.radius) +self.center[None][0], ti.random() * self.bottom +self.center[None][1] ]
            else:
                self.tube_x[i]=[ti.random() * (self.thick + self.radius) +self.center[None][0], ti.random() * self.bottom +self.center[None][1] ]

    #small functions
    def get_state(self):
        state=self.sample_v.to_numpy()
        return state
    
    def get_ball(self):
        #state=self.sample_v.to_numpy()
        return self.ball_center[None]
    #def get_grid(self):
    #    state=self.sample_v.to_numpy()
    #    return state

    def get_reward(self):
        #if self.ball_center[None][1] > 0.5:
        #    return 1
        #else:
        #    return -1
        return 10*math.exp( -( self.ball_center[None][1]-0.85 )*( self.ball_center[None][1]-0.85 )-( self.ball_center[None][0]-0.5 )*( self.ball_center[None][0]-0.5 ) ) + 2*math.exp(- self.ball_vel[None][0]*self.ball_vel[None][0]- self.ball_vel[None][1]*self.ball_vel[None][1]) 

    def get_rigid_state(self):
        II=np.array([self.center[None][0],self.center[None][1],self.angle[None]*0.1,self.vel[None][0]*0.1,self.vel[None][1]*0.1,self.omega[None]*0.1])
        return II

    def get_ball_state(self):
        ###self.cal_jelly_state()
        ###poss=np.zeros(2)#(10)
        #phos=np.zeros(10)
        ###vels=np.zeros(2)#(10)
        #Rs=np.zeros(5)
        a=np.zeros(4)
        a[0]=self.ball_center[None][0]
        a[1]=self.ball_center[None][1]
        a[2]=self.ball_vel[None][0]*0.1
        a[3]=self.ball_vel[None][1]*0.1

        ###for i in range(5):
        ###    k=i#self.o[i]
        ###    poss[2*i]=self.jelly_pos[2*k]
        ###    poss[2*i+1]=self.jelly_pos[2*k+1]
        ###    vels[2*i]=self.jelly_vel[2*k]
        ###    vels[2*i+1]=self.jelly_vel[2*k+1]
        ###for i in range(5):
        ###    Rs[i]=self.R[i]
        #poss[20]=0.025
        #print(Rs)
        #a=np.append(poss,vels)
        #return np.append(a,Rs)
        return a


if __name__=="__main__":
    ti.init(arch=ti.gpu, random_seed=int(time.time()))
    quality = 2.5
    E_ = 850
    mass = [50]
    rho = 3.4
    G = 20
    P = 1
    R = [5., 9., 3., 6., 4., 2., 8., 0., 7., 1.]
    S = [0.0325, 0.0, 0.06]
    ax = 0.
    av = 0.
    aw = 0.
    myMPM = MPMSolver(quality, E_, mass, rho, G, P, R, S)
    ###gui = ti.GUI("Taichi MLS-MPM-Tube-Ball-Liquid", res=512, background_color=0x112F41)
    color_select=np.array([0xff7fff,
                            0xff99ff,
                            0xffb2ff,
                            0xffd1ff,
                            0xfff4ff,0x84bf96,0x000000],dtype=np.uint32)
    myMPM.initialize()
    recorcor = 0
    #sample_grid = myMPM.get_state().reshape([128*128,2])
    #np.savetxt(str(frame+1)+"_sample.txt",sample_recording)
    #print(myMPM.color.to_numpy().max())
    for frame in range(2000000):
        recorcor += 1
        ax = 30.
        av = 0.
        aw = 600.

        #if gui.get_event(ti.GUI.PRESS):
        #    if gui.event.key == 'r':    
        #        myMPM.initialize()      
        #    elif gui.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
        #        break
        #if gui.is_pressed(ti.GUI.LEFT, 'a'):
        #    ax += -30.
        #if gui.is_pressed(ti.GUI.RIGHT, 'd'):
        #    ax += 30.
        ##if gui.is_pressed(ti.GUI.UP, 'w'):
        ##    av += 30.
        ##if gui.is_pressed(ti.GUI.DOWN, 's'):
        ##    av += -30.
        #if gui.is_pressed(ti.GUI.LMB):
        #    aw += 600.
        #if gui.is_pressed(ti.GUI.RMB):
        #    aw += -600.
        ##myMPM.penshui()
        ##for pen in range(int(2e-4 // myMPM.dt)):
        #    #myMPM.penshui()

        ax *= 1 if np.random.rand()>0.5 else -1
        aw *= 1 if np.random.rand()>0.5 else -1
        for s in range(int(2e-3 // myMPM.dt)):
            myMPM.penshui()
            myMPM.solve_tube(ax, av, aw)
            myMPM.substep()
            myMPM.solve_ball()
        #if myMPM.x.max > 
        #xtor = myMPM.tube_x.to_numpy()
        #xpr=np.empty((0, 2))
        #for xp in xtor:
            #print(xp)
            #if xp[1] > 0 and xp[0] > 0 and xp[0] < 1:
            #    np.vstack([xpr,xp])
        #tmp_ = myMPM.getX()
        ###gui.circles(myMPM.x.to_numpy(), radius=1.5, color=0x84bf96)
        ###gui.circles(myMPM.tube_x.to_numpy(), radius=1.5,color=0x845538)
        ###gui.circles(myMPM.ball_x.to_numpy(), radius=1.5,color=0xffb2ff)
        ###gui.show()
        if myMPM.get_ball()[1] < 0.15:
            myMPM.initialize()
        
        if recorcor == 30:
            recorcor = 0
            #print(myMPM.get_state())
            #ori_sample_recording = myMPM.get_state()
            sample_recording=np.transpose(myMPM.get_state(), (2, 0, 1))#.reshape([128*128,2])
            #print(sample_recording)
            np.savez("{}_sample".format(frame+1), data=sample_recording)
            #np.savetxt(str(frame+1)+"_sample.txt",sample_recording)




















