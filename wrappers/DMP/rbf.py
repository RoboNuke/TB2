from math import exp
import torch
import matplotlib.pyplot as plt

from wrappers.DMP.cs import CS

class RBF():
    def __init__(self, can_sys: CS, num_rbfs: int=8):
        self.num_rbfs = num_rbfs
        self.can_sys = can_sys
        self.centerRBFs()


    def eval(self, x):
        if type(x) == float:
            dx = x - self.cs
        else:
            dx = x[:,None] - self.cs[None, :]
        return torch.exp(-self.hs[None, :] * (dx ** 2))
    
    #def theeMat(self, x):
    #    mat = self.eval(x)
    #    return mat
    
    def centerRBFs(self):
        des_c = torch.linspace(0, 1.0, self.num_rbfs+2)[1:-1]
        self.cs = torch.exp(-self.can_sys.ax * des_c)
        self.hs = self.num_rbfs / (self.cs ** 2) 


    def centerRBFsV2(self):
        self.cs = torch.exp(-self.can_sys.ax * torch.linspace(0,1, self.num_rbfs))
        self.hs = torch.zeros_like(self.cs)
        for i in range(self.num_rbfs -1):
            self.hs[i] = 1 / ((self.cs[i] - self.cs[i]**2))
        self.hs[-1] = self.hs[-2]
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
    
if __name__=="__main__":
    import numpy as np
    #num_envs = 3
    num_rbf = 25
    ax = 5.0
    dt = 0.0001

    cs = CS(dt=dt)

    rbfs = RBF(cs, num_rbf)

    xs = cs.rollout()
    print(xs.size())
    ts = np.linspace(0, 1.0, int(1/dt))
    print(ts.shape)
    ys = rbfs.eval(xs)
    print(ys[:,:].size())
    plt.figure(1)
    plt.plot(ts, ys[:,:])
    plt.title("RBFs as function of Time")
    plt.figure(2)
    plt.plot(xs[:], ys[:,:])
    plt.title("RBFs as function of CS")
    plt.gca().invert_xaxis()
    plt.show()