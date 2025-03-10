from math import exp
import matplotlib.pyplot as plt
import numpy as np
from wrappers.DMP.rbf import RBF
from wrappers.DMP.cs import CS
import torch

class DiscreteDMP():
    def __init__(self, 
                 nRBF:int=100, 
                 betaY: float=1, 
                 dt: float= 0.001, 
                 cs: CS=CS(1.0, 0.001), 
                 num_envs: int=1, 
                 num_dims: int=1,
                 device: str="cpu"
        ):
        self.nRBF = nRBF
        self.ay = 4 * betaY
        self.by = betaY
        self.cs = cs
        self.dt = dt
        self.num_envs = num_envs
        self.num_dims = num_dims
        self.device = device

        self.ws = torch.ones((self.num_envs, self.nRBF, self.num_dims), device = self.device) #[1.0 for x in range(self.nRBF)]

        self.rbfs = RBF(can_sys=self.cs, num_rbfs=self.nRBF, device=self.device)

        self.ddy = 0
        self.dy = 0
        self.y = 0
        self.goal = None
        self.y0 = None


    def learnWeightsCSDT(self, y, dy, ddy, t):
        # expected input
        # y: position trajectory (num_envs, num_dims, timestep)
        # dy: velocity trajectory (num_envs, num_dims, timestep)
        # ddy: acceleration trajectory (num_envs, num_dims, timestep)
        # t: time for each above point (timestep)
        # points are assumed to be pl

        self.goal = y[:,-1,:].clone()
        x = self.cs.rollout()
        self.y0 = y[:,0,:].clone()
        
        rbMats = self.rbfs.eval(x) #[self.RBFs[i].theeMat(x) for i in range(self.nRBF)]
        #print("mats:", rbMats)
        #fd = ddy - self.ay * (self.by * (self.goal - y) - dy)
        #print(ddy/self.ay)
        #print((ddy/self.ay + dy) / self.by)
        a = (ddy/self.ay) + dy#) / self.ay 
        #print( (self.goal - self.y0).size(), x.size(), y.size(), self.goal.size())
        b = (self.goal - self.y0)[:,None,:] * x[None,:,None] + y - self.goal[:,None,:]
        #print("a,b:", a.size(), b.size())
        #fd = (ddy/self.ay + dy ) / self.by + (self.goal - self.y0) * x + y - self.goal
        fd = a/self.by + b

        #print("fd:", fd)
        s = x#[None,:,None] #* (self.goal - self.y0)[:,None,:]
        #print(type(s), type(rbMats), type(fd))
        #print(s.size(), rbMats.size(), fd.size())
        top = torch.sum(  (s[:,None] * rbMats)[None,:,:,None] * fd[:,:,None,:], dim=1)
        #print("top:", top.size())
        bot = torch.sum( s[:,None] * rbMats * s[:,None], dim=0)
        #print("top:", top.size())
        #print("bot:", bot.size())
        self.ws = top / bot[None,:,None]
        #print("ws pre:", self.ws)
        #print(self.goal.size(), self.y0.size())
        #print(self.goal, type(y))
        #print("y shape:", y.shape)
        #print("w shape:", self.ws.size())

        delta = self.goal[:,None,:]  - self.y0[:,None,:]
        delta = delta.repeat( (1,self.nRBF,1))
        delta_idx = delta > 0.00001
        self.ws[delta_idx] /= delta[delta_idx] 
        #if abs(delta) > 0.0001:
        #    self.ws /= delta
        #print("ws post:", self.ws)
        #self.ws[0] = 0.0

    def learnWeights(self,y, ydot, ydotdot, tt):
        self.goal = y[-1]
        x = self.cs.rollout()
        #dt = t / t[-1]
        path = torch.zeros((len(x)), device=self.device)
        dy = torch.zeros((len(x)), device=self.device)
        ddy = torch.zeros((len(x)), device=self.device)
        ts = torch.zeros(len(x), device=self.device)
        #t=torch.linspace(0, 1.0, len(y))
        
        
        # TODO Torch Interpolate
        import scipy.interpolate
        path_gen = scipy.interpolate.interp1d(tt, y)
        dy_gen = scipy.interpolate.interp1d(tt, ydot)
        ddy_gen = scipy.interpolate.interp1d(tt, ydotdot)
        import math
        for i in range(len(x)):
            #tx = min(1*np.log(x[i]) / -self.cs.ax, 1.0)
            tx = i * self.cs.dt
            print(tx)
            #$print(x[i], tx, i, i/len(x))
            path[i] = float(path_gen(tx))
            dy[i] = float(dy_gen(tx))
            ddy[i] = float(ddy_gen(tx))
            ts[i] = tx #i * self.dt
        #assert 1 ==0
        # estimate the gradients
        #ddt = self.dt * t[-1]
        # TODO Torch gradients
        #dy = np.gradient(path)#/(self.dt)
        #ddy = np.gradient(dy)#/(self.dt)

        #dy = torch.from_numpy(dy)
        
        #ddy = torch.from_numpy(ddy)
        """
        print(min(tt), max(tt), min(ts), max(ts))
        plt.plot(ts, path, '-r')
        plt.plot(tt/tt[-1], y)
        plt.show()
        plt.plot(ts ,dy, '-r')
        plt.plot(tt/tt[-1], ydot)
        plt.show()
        plt.plot(ts,ddy, '-r')
        plt.plot(tt/tt[-1], ydotdot)
        plt.show()
        """
        y = path
        
        self.y0 = y[0]
        #x = x.unsqueeze(1)
        # rbMats = (len(x), num_rbfs)
        rbMats = self.rbfs.eval(x) #[self.RBFs[i].theeMat(x) for i in range(self.nRBF)]

        fd = ddy - self.ay * (self.by * (self.goal - y) - dy)
        
        s = x #* (self.goal - self.y0)
        top = torch.sum(s[:,None] * rbMats * fd[:,None], dim=0)
        bot = torch.sum( s[:,None] * rbMats * s[:,None], dim=0)

        self.ws = top / bot

        if abs(self.goal  - y[0]) > 0.0001:
            self.ws /= (self.goal - y[0])

            

    def calcWPsi(self, x: float):
        thee = self.rbfs.eval(x) 
        #print(thee.size(), self.ws.size() )
        top =  torch.sum(thee[None,:,None] * self.ws, dim=1)
        bot = torch.sum(thee, dim=0)
        

        bot[bot <= 1e-6] = 1.0
        #print(bot)
        #print("calcWPsi tb:", top.size(), bot.size())
        #print( (top/bot).size())
        return top / bot
    
    def step(self, tau=1.0, error=0.0, external_force=None):
        ec = 1.0 / (1.0+error)

        x = self.cs.step(tau=tau, error_coupling = ec)
        
        #print(self.calcWPsi(x).size(), self.goal.size(), self.y0.size())
        F = self.calcWPsi(x) * x #(self.goal - self.y0) * x

        #self.ddy = self.ay * (self.by * (self.goal - self.y) - self.dy) + F
        self.ddy = self.ay * (self.by * (self.goal - self.y - (self.goal - self.y0) * x + F) - self.dy)

        if external_force != None:
            self.ddy += external_force
        
        self.dy += self.ddy * self.dt * ec / tau
        self.y += self.dy * self.dt * ec / tau

        #print(self.y.size(), self.dy.size(), self.ddy.size())

        return self.y, self.dy, self.ddy

    def reset(self, goal, y, dy = 0.0, ddy = 0.0):
        self.y = y.clone()
        self.dy = dy.clone()
        self.ddy = ddy.clone()
        self.y0 = self.y.clone()
        self.goal = goal.clone()
        self.cs.reset()

    def rollout(self, g, y0, dy0=0, ddy0=0, tau=1, scale=1):
        self.reset(g, y0, dy0, ddy0)
        t = 0.0

        # allocate memory
        timesteps = int(self.cs.timesteps * tau)
        z = torch.zeros( (self.num_envs, timesteps+1, self.num_dims), device=self.device)
        dz = torch.zeros_like(z)
        ddz = torch.zeros_like(dz)
        ts = torch.zeros(timesteps+1)
        #print(self.dt)
        ts[0] = t
        z[:,0,:] = self.y
        dz[:,0,:] = self.dy
        ddz[:,0,:] = self.ddy
        #print(f"Total Time:{self.cs.run_time * tau}")
        for it in range(timesteps):
            #t = np.log(self.cs.x) / -self.cs.ax
            t += self.cs.dt

            self.step(tau=tau, error=0.0, external_force=None)

            z[:,it+1,:] = self.y
            dz[:,it+1,:] = self.dy
            ddz[:,it+1,:] = self.ddy
            ts[it+1] = t

        return(ts, z, dz, ddz)


def plotSub(ax, t, ts, org, cpy, tit="DMP", ylab="Function"):
    ax.plot(t, org)
    ax.plot(ts, cpy,'r--')
    #ax.plot(np.linspace(0, t[-1], len(cpy)), cpy,'r--')
    ax.set(ylabel=ylab)
    ax.set_title(tit,  fontsize=20)

def single_dim_test():
    dt = 10.0/250.0
    for dt in [1/200.0, 10/200.0, 10/500, 1/500]:
        
        tmax = 1.0
        
        dmp = DiscreteDMP(nRBF=10, betaY=12.5/4.0, dt=dt, cs=CS(ax=2.5, dt=dt))

        t = np.arange(0, tmax, dt) / tmax
        
        of = 1.0
        ns = 0.1
        y1 = np.sin(of* 10*t) + np.cos(of * 3 *t) 
        y = y1 + np.random.normal(0, ns, t.shape)
        dy1 = of * 10*np.cos(of* 10*t) - of*3*np.sin(of*3*t) 
        dy = dy1 + np.random.normal(0, ns * 10, t.shape)
        ddy1 = -100* of**2 * np.sin(of*10*t) - 9 * of**2 * np.cos(of*3*t) 
        ddy = ddy1 + np.random.normal(0, ns * 100, t.shape)
        t *= tmax

        print("single:", y.shape, dy.shape, ddy.shape)
        dmp.learnWeightsCSDT(y,dy,ddy,torch.from_numpy(t))
        print("Weights:", dmp.ws)
        return
        tau = 1.0
        scale =1
        g = y[-1] * scale
        
        dmp.reset(y[0], dy[0], ddy[0])
        ts, z, dz, ddz = dmp.rollout(g, y[0], dy[0], ddy[0], tau, scale)
        
        fig, axs = plt.subplots(3)
        fig.set_figwidth(800/96)
        fig.set_figheight(1000/96)
        fig.tight_layout(pad=5.0)
        print(t[-1], ts[-1] )
        plotSub(axs[0], t, ts, y, z,f"Position DMP ({dt})", "Position")
        axs[0].plot(t, y1, 'g')
        plotSub(axs[1], t, ts, dy, dz, "Velocity DMP", "Velocity")
        axs[1].plot(t, dy1, 'g')
        plotSub(axs[2], t, ts, ddy, ddz, "Accel DMP", "Acceleration")
        axs[2].plot(t, ddy1, 'g')
        plt.xlabel("time (s)")

        plt.legend(['Noisy Function', 'Learned Trajectory', 'Original Function'])
    #plt.show()

    for j in range(10):
        g = np.random.rand()
        print("Goal:", g)
        for i in range(10):
            dmp.ws = np.random.normal(0, 10, dmp.ws.shape)
            dmp.reset(0, 0, 0)
            ts, z, dz, ddz = dmp.rollout(g=g, y0=0)
            plt.plot(ts, z)
        plt.show()


def test(num_dims = 1, num_envs = 1, noise = 0.1,
         tau=1.0, scale=1.0, tmax=1.0, dt=1/50):
    # define 3-D trajectories
    dmp = DiscreteDMP(
        nRBF=10, 
        betaY=12.5/4.0, 
        dt=dt, 
        cs=CS(ax=2.5, dt=dt/tmax), 
        num_envs=num_envs, 
        num_dims=num_dims
    )

    t = torch.linspace(start=0, end=tmax, steps= int(tmax/dt) ) 
    
    sco = torch.tensor([10.0, 3, 5])
    cco = torch.tensor([3,  6, 1.5])
    
    cco = cco.repeat(num_envs, 1)
    sco = sco.repeat(num_envs,1)
    for i in range(num_envs):
        if i > 2:
            cco[i,:] *= 0.25
            sco[i,:] *= 0.25
        else:
            cco[i,:] = torch.roll(cco[i,:], i)
    if num_dims < 3:
        sco = sco[:,:num_dims]
        cco = cco[:,:num_dims]
        
    y = torch.sin(sco[:,None,:] * t[None,:,None]) + torch.cos(cco[:,None,:] * t[None,:,None]) 
    y[-1,:,-1] = 3 # make an example with constant value
    dy = sco[:,None,:] * torch.cos(sco[:,None,:]*t[None,:,None]) - cco[:, None,:] * torch.sin(cco[:, None,:]*t[None, :,None])
    dy[-1,:,-1] = 0
    ddy = -sco[:, None,:]**2 * torch.sin(sco[:,None,:]*t[None,:,None]) - cco[:,None,:]**2 * torch.cos(cco[:,None,:]*t[None:,None])
    ddy[-1,:,-1] = 0
    #ddy *= 0
    #print(y.size(), dy.size(), ddy.size())

    fig, axs = plt.subplots(num_envs, num_dims)
    fig.set_figwidth(num_dims / 3 * 1600/96)
    fig.set_figheight(num_envs / 4 * 1000/96)
    fig.tight_layout(pad=5.0)

    dmp.learnWeightsCSDT(y, dy, ddy, t)
    #print(torch.max(dmp.ws), torch.min(dmp.ws), torch.mean(dmp.ws))
    #dmp.reset(y[:,0,:], dy[:,0,:], ddy[:,0,:])
    #print("goal:", y[:,-1,i])
    ts, z, dz, ddz = dmp.rollout(y[:,-1,:], y[:,0,:], dy[:,0,:], ddy[:,0,:], tau, scale)
    ts *= tmax
    #z = 0.95 * y
    if num_dims > 1 and num_envs > 1:
        for i in range(num_envs):
            for j in range(num_dims):
                axs[i,j].plot(t, y[i,:,j], label="Original")
                axs[i,j].plot(ts, z[i,:,j], 'r--', label="Fit DMP")
                axs[i,j].set_title(f"DMP (env={i}, dim={j})",  fontsize=20)
                axs[i,j].set(xlabel = 'Time (s)')
                axs[i,j].set(ylabel ='Position (m)')
    elif num_dims > 1 and num_envs == 1:
        for j in range(num_dims):
            axs[j].plot(t, y[0,:,j], label="Original")
            axs[j].plot(ts, z[0,:,j], 'r--', label="Fit DMP")
            axs[j].set_title(f"DMP (dim={j})",  fontsize=20)
            axs[j].set(xlabel = 'Time (s)')
            axs[j].set(ylabel ='Position (m)')
    elif num_dims == 1 and num_envs > 1:
        for i in range(num_envs):
            axs[i].plot(t, y[i,:,0], label="Original")
            axs[i].plot(ts, z[i,:,0], 'r--', label="Fit DMP")
            axs[i].set_title(f"DMP (env={i})",  fontsize=20)
            axs[i].set(xlabel = 'Time (s)')
            axs[i].set(ylabel ='Position (m)')
    else:
        axs.plot(t, y[0,:,0], label="Original")
        axs.plot(ts, z[0,:,0], 'r--', label="Fit DMP")
        axs.set_title(f"DMP",  fontsize=20)
        axs.set(xlabel = 'Time (s)')
        axs.set(ylabel ='Position (m)')

    plt.legend()
    plt.show()

if __name__=="__main__":
    #single_dim_test()
    #multi_dim_test()
    test(num_dims=3,num_envs=4, tmax=0.5)
    assert 1 == 0
    for i in [1,4]:
        for j in [1,3]:
            test(num_dims=j,num_envs=i)