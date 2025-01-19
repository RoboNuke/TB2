import numpy as np
import matplotlib.pyplot as plt

class CS():
    def __init__(self, ax, dt):
        self.run_time = 1.0
        self.dt = dt
        self.timesteps = int(self.run_time / self.dt)
        self.xPath = []
        self.ax = ax
        self.x = self.run_time

    def reset(self):
        self.x = self.run_time

    def step(self, tau=1.0, error_coupling=1.0):
        self.x += (-self.ax * self.x * error_coupling) * self.dt / tau
        return self.x

    def rollout(self,tau=1.0):
        self.reset()
        self.xPath = [0 for i in range(self.timesteps)]
        for t in range(self.timesteps):
            self.xPath[t] = self.x
            self.step(tau=tau)
        return self.xPath
    
if __name__=="__main__":
    cs = CS(5.0, 0.01)
    path = cs.rollout(100)
    t = np.linspace(0,1,len(path))
    plt.plot(t, path)
    plt.show()