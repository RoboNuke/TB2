from math import exp
import torch
import matplotlib.pyplot as plt


class RBF():
    def __init__(self, cs, hs):
        self.c = cs
        self.h = hs

    def eval(self, x):
        dx = x[:,None] - self.c[None, :]
        return torch.exp(-self.h[None, :] * (dx ** 2))
    
    def theeMat(self, t):
        mat = np.zeros(len(t))
        for i in range(len(t)):
            mat[i] = self.eval(t[i])

        return mat
    
if __name__=="__main__":
    import numpy as np
    #num_envs = 3
    num_rbf = 25
    ax = 5.0
    dt = 0.0001
    centers = torch.zeros( (num_rbf))
    std_devs = torch.zeros( (num_rbf))

    des_c = np.linspace(0.0, 1.0, num_rbf+2)[1:-1]
    #print(len(self.cs.xPath))
    for i in range(num_rbf):
        c = np.exp(-ax * (des_c[i]))
        #c = des_c[i]
        #h = 25
        h = (num_rbf) / (c ** 2)
        #h = num_rbf ** 1.5 / c / ax
        centers[i] = c
        std_devs[i] = h

    rbfs = RBF(centers, std_devs)

    from wrappers.DMP.cs import CS
    cs = CS(dt=dt)

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