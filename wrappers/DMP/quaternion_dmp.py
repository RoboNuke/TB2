from math import exp
from wrappers.DMP.rbf import RBF
from wrappers.DMP.cs import CS
import torch

class QuaternionDMP():
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

        self.ws = torch.ones((self.num_envs, 3, self.nRBF), device = self.device)
        self.rbfs = RBF(can_sys=self.cs, num_rbfs=self.nRBF)

        self.q = torch.zeros((self.num_envs, 4), device = self.device)
        self.n = torch.zeros((self.num_envs, 3), device = self.device)
        self.dn = torch.zeros((self.num_envs, 3), device = self.device)
        self.gq = torch.zeros((self.num_envs, 4), device = self.device)

    def cross(self, a, b):
        # a/b: (num_envs, timestep, 3)
        a1 = a[:,:,0]
        a2 = a[:,:,1]
        a3 = a[:,:,2]
        b1 = b[:,:,0]
        b2 = b[:,:,1]
        b3 = b[:,:,2]

        cros = torch.zeros_like(b)
        cros[:,:,0] = a2 * b3 - a3 * b2
        cros[:,:,1] = a3 * b1 - a1 * b3
        cros[:,:,2] = a1 * b2 - a2 * b1
        return cros
    
    def singleCross(self, a, b):
        # a/b: (num_envs, 3)
        a1 = a[:,0]
        a2 = a[:,1]
        a3 = a[:,2]
        b1 = b[:,0]
        b2 = b[:,1]
        b3 = b[:,2]

        cros = torch.zeros_like(b)
        cros[:,0] = a2 * b3 - a3 * b2
        cros[:,1] = a3 * b1 - a1 * b3
        cros[:,2] = a1 * b2 - a2 * b1
        return cros


    def quatDiff(self, a, b):
        # quaternion multiplies a * conj(b)
        # a and b: (num_envs, timestep, 4) (w,x,y,z) ordering
        #conj_b = torch.zeros_like(b)
        conj_b = torch.clone(b)
        if len(conj_b.size()) == 3:
            conj_b[:,:,1:] *= -1
            return self.multq(a, conj_b)    
        else:
            conj_b[:,1:] *= -1
            return self.singleMultq(a, conj_b)


    def wMultq(self, a, b):
        delta = torch.zeros_like(b)
        v1 = 0.0
        u1 = a
        v2 = b[:,0]
        u2 = b[:,1:]
        delta[:,0] = v1 * v2 - torch.sum(u1 * u2, dim=1)
        delta[:,1:] = v1 * u2 + v2 * u1 + self.singleCross(u1, u2)
        delta /= torch.linalg.norm(delta, dim=1)
        return delta


    def singleMultq(self, a, b):
        delta = torch.zeros_like(b)
        v1 = a[:,0]
        u1 = a[:,1:]
        v2 = b[:,0]
        u2 = b[:,1:]
        delta[:,0] = v1 * v2 - torch.sum(u1 * u2, dim=1)
        delta[:,1:] = v1 * u2 + v2 * u1 + self.singleCross(u1, u2)
        delta /= torch.linalg.norm(delta, dim=1)
        return delta

    def multq(self, a, b):
        delta = torch.zeros_like(b)
        v2 = b[:,:,0]
        u2 = b[:,:,1:]
        if len(a.size()) == 3:
            v1 = a[:,:,0]
            u1 = a[:,:,1:]
            delta[:,:,0] = (v1 * v2) - torch.sum(u1*u2,dim=2)
            delta[:,:,1:] = v1[:,:,None] * u2 + v2[:,:,None] * u1 + self.cross(u1, u2)
        else:
            v1 = a[:,0]
            u1 = a[:,1:]
            delta[:,:,0] = v1[:,None] * v2 - torch.sum(u1 * u2, dim=2)
            delta[:,:,1:] = v1[:,None,None] * u2 + v2[:,:,None] * u1[:,None,:] + self.cross(u1[:,None,:], u2)
        delta /= torch.linalg.norm(delta, dim=2)[:,:,None]
        return delta

    def logq(self, a):
        # performs quaternion log on a 
        # a: (num_envs, timestep, 4) (w,x,y,z) q ordering
        #print("a", a.size())
        if len(a.size()) == 3:
            log_out = torch.zeros((a.size()[0], a.size()[1], 3), device=self.device)
            u_norm = torch.linalg.norm(a[:,:,1:], dim=2) 
            #print(u_norm)
            log_out[:,:,:] = torch.acos(a[:,:,0])[:,:,None] * a[:,:,1:] / u_norm[:,:,None]
            log_out[u_norm < 1e-2,:] = torch.tensor([0,0,0], device=self.device, dtype=a.dtype)
        else:
            log_out = torch.zeros((a.size()[0], 3), device=self.device)
            u_norm = torch.linalg.norm(a[:,1:], dim=1)
            #print(u_norm)
            log_out[:,:] = torch.acos(a[:,0])[:,None] * a[:,1:] / u_norm[:,None]
            log_out[u_norm < 1e-6,:] = torch.tensor([0,0,0], device=self.device, dtype=a.dtype)

        #print(log_out)
        return log_out
    
    def expq(self, a):
        exp_out = torch.zeros_like(a)
        w_norm = torch.linalg.norm(a, dim=1)
        #w_norm[w_norm > 3.14159] -= 3.14159
        exp_out[:,0] = torch.cos(w_norm)
        exp_out[:,1:] = torch.sin(w_norm) * a[:,1:] / w_norm
        exp_out[w_norm < 1e-6,0] = 1.0
        exp_out[w_norm < 1e-6,1:] = 0.0
        #print("exp out size:", exp_out.size())
        return exp_out

    def learnWeightsCSDT(self, q, n, dn, t):
        # expected input
        # q: quaternion trajectory (num_envs, 4, timestep)
        # n: angular velocity (quaternion with zero complex term) (num_envs, 4, timestep)
        # dn: angular acceleration trajectory (quaternion with zero complex term) (num_envs, 4, timestep)
        # t: time for each above point (timestep)

        self.goal = q[:, -1, :]
        x = self.cs.rollout()
        
        self.q0 = q[:, 0, :]

        rbMats = self.rbfs.eval(x) #(num_envs, direction, num_rbfs) 
        #print(self.logq(self.quatDiff(self.goal, q)).size(), dn.size(), n.size())
        fd = dn - self.ay * ( 2 * self.by * self.logq(self.quatDiff(self.goal, q)) - n)
        #fd /= (2 * self.logq(self.quatDiff(self.goal, self.q0)))
        #assert 1 == 0
        s = x[None, :, None] / (2 * self.logq(self.quatDiff(self.goal, self.q0)))[:,None,:]
        # top becomes (num_envs, timesteps, direction, weight)

        for env_idx in range(self.num_envs):
            for i in range(self.nRBF):
                for k in range(3):
                    sm = s[env_idx,:,k]
                    rb = rbMats[:, k]
                    f = fd[env_idx, :, k]
                    tt = torch.sum(sm * rb[None,:] * f)
                    bt = torch.sum(sm * rb[None,:] * sm)
                    self.ws[env_idx,k] = tt/bt
            
            

        #top = torch.sum(s[None,:,None,None] * rbMats[None,:,None,:] * fd[:,:,:,None], dim=1)
        #bot = torch.sum( s[None,:,None,None] * rbMats[None,:,None,:] * s[None,:,None,None], dim=1)

        #self.ws = top / bot
        

    def calcWPsi(self, x: float):
        thee = self.rbfs.eval(x) #(num_rbfs) 
        top =  torch.sum(thee[None, None, :] * self.ws, dim=2)
        bot = torch.sum(thee)
        return top / bot
    
    def step(self, tau=1.0, error=0.0, external_force=None):
        ec = 1.0 / (1.0+error)

        x = self.cs.step(tau=tau, error_coupling = ec)
        
        F = 2*self.logq(self.quatDiff(self.goal, self.q)) * self.calcWPsi(x) * x 
        
        self.dn = self.ay * ( 2*self.by * self.logq(self.quatDiff(self.goal, self.q)) - self.n) + F
        
        self.n += self.dn * self.dt * ec / tau
        self.q = self.wMultq(self.expq(self.n * self.dt / (2 * tau)), self.q)
        self.q[self.q[:,0] < 0] *= -1
        return self.q, self.n, self.dn

    def reset(self, goal, q: torch.tensor, n = None, dn = None):
        self.q = q
        if n is None:
            self.n = torch.zeros_like(self.q)
        else:
            self.n = n
        if dn is None:
            self.dn = torch.zeros_like(self.q)
        else:
            self.dn = dn
        self.q0 = torch.clone(self.q)
        self.goal = goal
        self.cs.reset()

    def rollout(self, g, q0, n0=None, dn0=None, tau=1, scale=1):
        self.reset(g, q0, n0, dn0)
        t = 0.0
        #print(self.dt)
        ts = [0.0]
        #print(f"Total Time:{self.cs.run_time * tau}")
        timesteps = int(self.cs.timesteps * tau)
        z = torch.zeros((self.num_envs, timesteps+1, 4))
        w = torch.zeros((self.num_envs, timesteps+1, 3))
        dw = torch.zeros((self.num_envs, timesteps+1, 3))
        #print(q0, n0, dn0)
        z[:,0,:] = q0
        w[:,0,:] = n0
        dw[:,0,:] = dn0

        for it in range(timesteps):
            #t = np.log(self.cs.x) / -self.cs.ax
            t += self.cs.dt
            #print(self.q)
            self.step(tau=tau, error=0.0, external_force=None)
            #print(self.q)
            #assert 1 == 0
            #print(self.q, "\t", torch.linalg.norm(self.q))
            z[:,it+1, :] = self.q
            w[:,it+1, :] = self.n
            dw[:,it+1, :] = self.dn
            ts.append(t)

        return(ts, z, w, dw)

def pltqndn(fig, axs, t, data_ref, num_envs=1, names=['x', 'y', 'z', 'w'], colors=['r-','g-' ,'b-','m-']):
    
    for env_id in range(num_envs):
        for j in range(len(names)):
            for i in range(len(data_ref)):
                idx = j
                if names[j] == 'w' and i > 0:
                    continue
                if i == 0 and names[j] != 'w':
                    idx+=1
                elif i == 0 and names[j] == 'w':
                    idx = 0
                if names[j] in ['x','y','z']:
                    continue
                print(names[j], colors[j])
                if env_id > 1:
                    axs[i, env_id].plot(t, data_ref[i][env_id,:,idx], color=colors[j][0], linestyle=colors[j][1])
                else:
                    axs[i].plot(t, data_ref[i][env_id,:,idx], color=colors[j][0], linestyle=colors[j][1], label=names[j])
                #if colors[j][0] == 'k':
                #    plt.show()
    #fig.legend(names)


    return fig, axs


if __name__=="__main__":

    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.spatial.transform import Rotation

    dt = 1/500
    tmax = 1.0
        
    dmp = QuaternionDMP(nRBF=10, betaY=1/12, dt=dt, cs=CS(ax=0.5, dt=dt))

    t = torch.linspace(start=0.0, end=tmax, steps= int(tmax/dt) ) #/ tmax
    
    of = 1.0
    ns = 0.1
    beta = torch.tensor(3.14159/2)
    omega = torch.tensor(5.0)

    q = torch.zeros(1, len(t), 4)
    n = torch.zeros(1, len(t), 3)
    dn = torch.zeros_like(n)

    """
    q[:,:,1] = torch.sin(of* 10*t) #+ torch.cos(of * 3 *t) # set the "x" component of the quaternion

    # normalize the quaternion
    q[:, :, 0] = torch.sqrt(1 - q[:,:,1] ** 2)
    
    assert torch.all(torch.linalg.norm(q, dim=2) - 1 < 1e-6), "q not normalized"
    # fill n with dq/dt
    n[:,:,1] = of * 10 * torch.cos(of * 10 * t) #- of * 3 * torch.sin(of*3*t) 
    #calculate actual n
    n = 2 * dmp.quatDiff(n, q)
    print(n)
    # get dn with similar procedure
    dn[:,:,1] = -100* of**2 * torch.sin(of*10*t) #- 9 * of**2 * torch.cos(of*3*t) 
    #dn[:,:,0] = - dn[:,:,0]
    dn = 2 * dmp.quatDiff(dn, q)
    """
    q[:,:,0] = torch.cos(beta/2.0)
    q[:,:,1] = torch.sin(beta/2.0) * torch.cos(omega * t) 
    q[:,:,2] = torch.sin(beta/2.0) * torch.sin(omega * t) 
    q[:,:,3] = 0.0

    n[:,:,0] = -omega * torch.sin(beta) * torch.sin(omega * t) # / (omega**2 + 1 - 2 * torch.cos(beta))
    n[:,:,1] = omega * torch.sin(beta) * torch.cos(omega * t) #/ (omega**2 + 1 - 2 * torch.cos(beta))
    n[:,:,2] = omega * (torch.cos(beta)-1) #/ (omega**2 + 1 - 2 * torch.cos(beta))
    

    dn[:,:,0] = - (omega**2) * torch.sin(beta) * torch.cos(omega * t) 
    dn[:,:,1] = - (omega**2) * torch.sin(beta) * torch.sin(omega * t) 
    dn[:,:,2] = 0.0
    #print(n[:,:,0])
    #assert torch.all(n[:,:,0] < 1e-6), "Multiplaction not correct"
    #assert torch.all(dn[:,:,0] < 1e-6), 'accel mujlt not correct'
    
    #t *= tmax

    fig, axs = plt.subplots(3, 1)
    fig.set_figwidth(800/96)
    fig.set_figheight(1000/96)
    fig.tight_layout(pad=5.0)
    fig, axs = pltqndn(fig, axs, t, data_ref=[q, n, dn])
    #plt.show()

    #dmp.reset(q[:,-1,:], q, n, dn)
    dmp.learnWeightsCSDT(q,n,dn,t)
    
    tau = 1.0
    scale =1
    g = q[:,-1,:] * scale
    
    
    ts, z, w, dw = dmp.rollout(g, q[:,0,:], n[:,0,:]*0.0, dn[:,0,:]*0.0, tau, scale)
    
    print(z.size(), w.size(), dw.size())
    for i in [0,1,2,3]:
        fig = plt.figure(2)
        plt.plot(ts, z[0,:,i], 'k')
        plt.plot(t, q[0,:,i], 'r')
        plt.show()
    fig, axs = pltqndn(fig, axs, ts, data_ref=[z], colors=['k-' for i in range(4)]) #colors=['r:','g:' ,'b:','m-'])
    handles, labels = axs[0].get_legend_handles_labels()
    #labels[4] = 'Est'
    #fig.legend(handles[:5], labels[:5])
    #plotSub(axs[0], t, ts, y, z,f"Position DMP ({dt})", "Position")
    #axs[0].plot(t, y1, 'g')
    #plotSub(axs[1], t, ts, dy, dz, "Velocity DMP", "Velocity")
    #axs[1].plot(t, dy1, 'g')
    #plotSub(axs[2], t, ts, ddy, ddz, "Accel DMP", "Acceleration")
    #axs[2].plot(t, ddy1, 'g')
    plt.xlabel("time (s)")

    #plt.legend(['Noisy Function', 'Learned Trajectory', 'Original Function'])
    plt.show()
    """
    for j in range(10):
        g = 25 * np.random.rand()
        for i in range(10):
            dmp.ws = np.random.normal(0, 100, dmp.ws.shape)
            #dmp.reset(0, 0, 0)
            ts, z, dz, ddz = dmp.rollout(g=g, y0=0)
            plt.plot(ts, z)
        #plt.show()
    """