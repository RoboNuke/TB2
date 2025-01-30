from math import exp
import matplotlib.pyplot as plt
#import numpy as np
from wrappers.DMP.rbf import RBF
from wrappers.DMP.cs import CS
import torch

class DiscreteDMP():
    def __init__(self, nRBF=100, betaY=1, dt= 0.001, cs=CS(1.0, 0.001), num_envs=1, device="cpu"):
        self.nRBF = nRBF
        self.ay = 4 * betaY
        self.by = betaY
        self.cs = cs
        self.dt = dt
        self.num_envs = num_envs
        self.device = device

        self.ws = ws
        if self.ws == None:
            self.ws = torch.ones((self.num_envs, self.nRBF), device = self.device) #[1.0 for x in range(self.nRBF)]

        self.ddy = 0
        self.dy = 0
        self.y = 0
        self.goal = None
        self.y0 = None

    def centerRBFs(self):
        self.rbf_centers = torch.linspace((self.nRBF), device = self.device)
        self.rbf_std_dev = torch.linspace((self.nRBF), device = self.device) 
        des_c = torch.linspace(0, 1.0, self.nRBF)
        for i in range(self.nRBF):
            self.rbf_centers[i] = torch.exp(-self.cs.ax * des_c[i])
            #c = des_c[i]
            self.rbf_std_dev[i] = (self.nRBF) / (self.rbf_centers[i] ** 2)
            #h = self.nRBF ** 1.5 / c / self.cs.ax
            
            #c = np.exp(-self.cs.ax * (i - 1)/(self.nRBF - 1))

    """
    def centerRBFsV2(self):
        self.RBFs = []
        c = [np.exp(-self.cs.ax * i/self.nRBF) for i in range(self.nRBF)]
        h = [1 / ((c[i+1] - c[i])**2) for i in range(self.nRBF-1)]
        h.append(h[-1])
        #print(c)
        #print(h)
        for i in range(self.nRBF):
            self.RBFs.append(RBF(c[i],h[i]))
    """

    def learnWeights(self,y): #, ydot, ydotdot, tt):
        self.goal = y[-1]
        x = np.array(self.cs.rollout())
        #dt = t / t[-1]
        path = np.zeros(len(x))
        ts = np.zeros(len(x))
        t=np.linspace(0, self.cs.run_time, len(y))
        import scipy.interpolate
        path_gen = scipy.interpolate.interp1d(t, y)
        for i in range(len(x)):
            path[i] = path_gen(i * self.cs.dt)
            ts[i] = i * self.dt

        
        # estimate the gradients
        #ddt = self.dt * t[-1]
        dy = np.gradient(path)/(self.dt)
        ddy = np.gradient(dy)/(self.dt)
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
        rbMats = [self.RBFs[i].theeMat(x) for i in range(self.nRBF)]

        fd = ddy - self.ay * (self.by * (self.goal - y) - dy)
        
        for i in range(len(self.ws)):
            bot = np.sum( x ** 2 * rbMats[i])
            #bot = np.sum(x * rbMats[i])
            top = np.sum( x * rbMats[i] * fd)
            #print(bot)
            self.ws[i] = top / bot
        #print(g - y[0])
        if abs(self.goal - y[0]) > 0.0001:
            for i in range(self.nRBF):
                self.ws[i]/= (self.goal-y[0])

    def calcWPsi(self, x):
        top = 0
        bot = 0
        for i in range(len(self.ws)):
            thee = self.RBFs[i].eval(x)
            top += thee * self.ws[i] 
            bot += thee
        if bot > 1e-6:
            return top/bot
        else:
            return top
    
    def step(self, tau=1.0, error=0.0, external_force=None):
        ec = 1.0 / (1.0+error)

        x = self.cs.step(tau=tau, error_coupling = ec)

        F = self.calcWPsi(x) * (self.goal - self.y0) * x

        self.ddy = self.ay * (self.by * (self.goal - self.y) - self.dy) + F

        if external_force != None:
            ddy += external_force
        
        self.dy += self.ddy * self.dt * ec / tau
        self.y += self.dy * self.dt * ec / tau

        return self.y, self.dy, self.ddy

    def reset(self, goal, y, dy = 0.0, ddy = 0.0):
        self.y = y
        self.dy = dy
        self.ddy = ddy
        self.y0 = self.y
        self.goal = goal
        self.cs.reset()

    def rollout(self, g, y0, dy0=0, ddy0=0, tau=1, scale=1):
        self.reset(g, y0, dy0, ddy0)
        t = 0.0
        z = [y0]
        dz = [dy0]
        ddz = [ddy0]
        #print(self.dt)
        ts = [0.0]
        #print(f"Total Time:{self.cs.run_time * tau}")
        timesteps = int(self.cs.timesteps * tau)
        for it in range(timesteps):
            t = it * self.dt

            self.step(tau=tau, error=0.0, external_force=None)

            z.append(self.y)
            dz.append(self.dy)
            ddz.append(self.ddy)
            ts.append(t)

        z = np.array(z)
        dz = np.array(dz)/tau
        #dz[0]*=tau
        ddz = np.array(ddz)/(tau**2)
        #ddz[0]*=tau**2
        return(ts, z, dz, ddz)


def plotSub(ax, t, ts, org, cpy, tit="DMP", ylab="Function"):
    ax.plot(t, org)
    ax.plot(ts, cpy,'r--')
    #ax.plot(np.linspace(0, t[-1], len(cpy)), cpy,'r--')
    ax.set(ylabel=ylab)
    ax.set_title(tit,  fontsize=20)

if __name__=="__main__":
    dt = 0.001
    tmax = 5

    dmp = DiscreteDMP(nRBF=75, betaY=25.0/4.0, dt=dt, cs=CS(1.0, dt))

    t = np.arange(0, tmax, dt)
    of = 0.5
    y = np.sin(of* 10*t) + np.cos(of * 3 *t)
    dy = of * 10*np.cos(of* 10*t) - of*3*np.sin(of*3*t)
    ddy = -100* of**2 * np.sin(of*10*t) - 9 * of**2 * np.cos(of*3*t)

    dmp.learnWeights(y) #,dy,ddy,t)

    tau = 10
    scale = 1
    g = y[-1] * scale

    ts, z, dz, ddz = dmp.rollout(g, y[0], dy[0]*tmax, ddy[0]*tmax**2, tau, scale)
    
    fig, axs = plt.subplots(3)
    fig.set_figwidth(800/96)
    fig.set_figheight(1000/96)
    fig.tight_layout(pad=5.0)

    plotSub(axs[0], t*tau/tmax, ts, y, z,"Position DMP", "Position")
    plotSub(axs[1], t*tau/tmax, ts, dy*(tmax/tau), dz, "Velocity DMP", "Velocity")
    plotSub(axs[2], t*tau/tmax, ts, ddy*( (tmax/tau)**2), ddz, "Accel DMP", "Acceleration")
    plt.xlabel("time (s)")

    plt.legend(['Original Function', 'Learned Trajectory'])
    plt.show()