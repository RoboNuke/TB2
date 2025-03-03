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
                 device: str="cpu"
        ):
        self.nRBF = nRBF
        self.ay = 4 * betaY
        self.by = betaY
        self.cs = cs
        self.dt = dt
        self.num_envs = num_envs
        self.device = device

        self.ws = torch.ones((self.num_envs, self.nRBF), device = self.device) #[1.0 for x in range(self.nRBF)]

        self.rbfs = RBF(can_sys=self.cs, num_rbfs=self.nRBF)

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

        #print("init y:", y.shape)
        self.goal = y[-1]
        #print(self.goal.shape)
        x = self.cs.rollout()
        #print("x:", x)
        self.y0 = y[0]
        #x = x.unsqueeze(1)
        # rbMats = (len(x), num_rbfs)
        rbMats = self.rbfs.eval(x) #[self.RBFs[i].theeMat(x) for i in range(self.nRBF)]
        #print("mats:", rbMats)
        #fd = ddy - self.ay * (self.by * (self.goal - y) - dy)
        #print(ddy/self.ay)
        #print((ddy/self.ay + dy) / self.by)
        try:
            a = ddy / self.ay + dy 
        except:
            a = torch.from_numpy(ddy/self.ay + dy)
        b = (self.goal - self.y0) * x + y - self.goal
        #fd = (ddy/self.ay + dy ) / self.by + (self.goal - self.y0) * x + y - self.goal
        try:
            fd = a/self.by + b
            print("a try:", a.size())
        except:
            a = torch.from_numpy(ddy/self.ay + dy)
            print("a except:", a.size())
            fd = a/self.by + b
        #print("fd:", fd)
        s = x #* (self.goal - self.y0)
        print(type(s), type(rbMats), type(fd))
        print(s.size(), rbMats.size(), fd.size())
        try:
            top = torch.sum(s[:,None] * rbMats * fd[0,:,None], dim=0)
        except:
            top = torch.sum(s[:,None] * rbMats * fd[:,None], dim=0)
        bot = torch.sum( s[:,None] * rbMats * s[:,None], dim=0)
        #print("top:", top)
        #print("bot:", bot)
        self.ws = top / bot
        #print("ws pre:", self.ws)
        print(self.goal, type(y))
        print("y shape:", y.shape)
        if abs(self.goal  - y[0]) > 0.0001:
            self.ws /= (self.goal - y[0])
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
        top =  torch.sum(thee * self.ws[None,:], dim=1)
        bot = torch.sum(thee, dim=0)
        

        bot[bot <= 1e-6] = 1.0
        return top / bot
    
    def step(self, tau=1.0, error=0.0, external_force=None):
        ec = 1.0 / (1.0+error)

        x = self.cs.step(tau=tau, error_coupling = ec)
        
        F = self.calcWPsi(x) * (self.goal - self.y0) * x

        #self.ddy = self.ay * (self.by * (self.goal - self.y) - self.dy) + F
        self.ddy = self.ay * (self.by * (self.goal - self.y - (self.goal - self.y0) * x + F) - self.dy)

        if external_force != None:
            self.ddy += external_force
        
        self.dy += self.ddy * self.dt * ec / tau
        self.y += self.dy * self.dt * ec / tau

        #print(self.y.size(), self.dy.size(), self.ddy.size())

        return self.y, self.dy, self.ddy

    def reset(self, goal, y, dy = 0.0, ddy = 0.0):
        self.y = y
        self.dy = dy
        self.ddy = ddy
        self.y0 = self.y
        self.goal = goal
        self.cs.reset()

    def rollout(self, g, y0, dy0=0, ddy0=0, tau=1, scale=1):
        self.reset(g, y0.item(), dy0.item(), ddy0.item())
        t = 0.0
        print("Rollout start:", y0)
        z = [y0.item()]
        print(z)
        dz = [dy0.item()]
        ddz = [ddy0.item()]
        #print(self.dt)
        ts = [0.0]
        #print(f"Total Time:{self.cs.run_time * tau}")
        timesteps = int(self.cs.timesteps * tau)
        for it in range(timesteps):
            #t = np.log(self.cs.x) / -self.cs.ax
            t += self.cs.dt

            self.step(tau=tau, error=0.0, external_force=None)

            z.append(self.y.item())
            dz.append(self.dy.item())
            ddz.append(self.ddy.item())
            ts.append(t)

        z = np.array(z)
        dz = np.array(dz)#/tau
        #dz[0]*=tau
        ddz = np.array(ddz)#/(tau**2)
        #ddz[0]*=tau**2
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


def multi_dim_test():
    # define 3-D trajectories
    dt = 1/500
    tmax = 1.0
    tau = 1.0
    scale = 1.0
        
    dmp = DiscreteDMP(nRBF=10, betaY=12.5/4.0, dt=dt, cs=CS(ax=2.5, dt=dt))

    t = torch.linspace(start=0.0, end=tmax, steps= int(tmax/dt) ) #/ tmax

    y = torch.zeros(1, len(t), 3)
    dy = torch.zeros_like(y)
    ddy = torch.zeros_like(dy)
    
    sco = torch.tensor([10, 3, 5])
    cco = torch.tensor([3,  6, 1.5])

    y[0,:,:] = torch.sin(sco[None,:] * t[:,None]) + torch.cos(cco[None,:] * t[:,None]) 
    dy[0,:,:] = sco[None,:] * torch.cos(sco[None,:]*t[:,None]) - cco[None,:] * torch.sin(cco[None,:]*t[:,None])
    ddy[0,:,:] = -sco[None,:]**2 * torch.sin(sco[None,:]*t[:,None]) - cco[None,:]**2 * torch.cos(cco[None,:]*t[:,None])

    t *= tmax

    fig, axs = plt.subplots(3)
    fig.set_figwidth(800/96)
    fig.set_figheight(1000/96)
    fig.tight_layout(pad=5.0)

    for i in range(3):

        dmp.learnWeightsCSDT(y[0,:,i],dy[0,:,i],ddy[0,:,i],t)
        dmp.reset(y[0,0,i], dy[0,0,i], ddy[0,0,i])
        print("goal:", y[:,-1,i])
        ts, z, dz, ddz = dmp.rollout(y[0,-1,i], y[0,0,i], dy[0,0,i], ddy[0,0,i], tau, scale)
        axs[i].plot(t, y[0,:,i])
        axs[i].plot(ts, z)
        #axs[i].plot(t, dy1[0,:,i])
        #axs[i].plot(t, ddy1[0,:,i])
    plt.show()

if __name__=="__main__":
    single_dim_test()
    multi_dim_test()