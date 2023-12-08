import torch
import math
from func_defs import psi_func_torch

bias_bool = True

# ===================
# Network definitions
# ===================

class DiscNet(torch.nn.Module):

    def __init__(self, dim, ns, act_func, h, device, TT=1.0):
        super().__init__()
        self.lin1 = torch.nn.Linear(dim+1, ns, bias=bias_bool)
        self.lin2 = torch.nn.Linear(ns, ns, bias=bias_bool)
        self.lin3 = torch.nn.Linear(ns, ns, bias=bias_bool)
        self.linlast = torch.nn.Linear(int(ns), 1)
        self.act_func = act_func

        self.dim = dim
        self.h = h
        self.TT = TT
        self.device = device

    def forward(self, t, inp):
        t_normalized = t - self.TT/2

        out = torch.cat((t_normalized, inp), dim=1)
        out = self.act_func(self.lin1(out))

        out = self.act_func(out + self.h * self.lin2(out))
        # out = self.act_func(self.lin2(out))
        out = self.act_func(out + self.h * self.lin3(out))
        # out = self.act_func(self.lin3(out))

        out = self.linlast(out)
        out = out

        ctt = t.view(inp.size(0), 1)
        c1 = (self.TT - ctt) / self.TT  # convex weight 1
        c2 = ctt / self.TT  # convex weight 2

        # 可恢复 return c1 * out \
        # 可恢复     + c2 * psi_func_torch(inp, self.device).view(inp.size(0), 1)
        return c1*out

class GenNet(torch.nn.Module):

    def __init__(self, dim, ns, act_func, h, device, mu, std, TT=1.0):
        super().__init__()
        self.mu = mu
        self.std = std
        self.lin1 = torch.nn.Linear(dim+1, ns)
        self.lin2 = torch.nn.Linear(ns, ns)
        self.lin3 = torch.nn.Linear(ns, ns)
        self.linlast = torch.nn.Linear(int(ns), dim)
        self.act_func = act_func

        self.dim = dim
        self.h = h
        self.TT = TT
        self.device = device

    def forward(self, t, inp, samples_from_Terminal_set):
        # 我的
        inp_normalized = torch.zeros_like(inp)
        for i in range(inp.shape[1]):
            if self.std[i] == 0:
                inp_normalized[:,i] = self.mu[i] + 1.0*torch.rand(1,1)
            else:
                inp_normalized[:,i] = (inp[:,i] - self.mu[i])*(1 / self.std[i])
        # 可恢复inp_normalized = (inp - self.mu.expand(inp.size())) * (1 / self.std.expand(inp.size()))
        t_normalized = t - self.TT/2

        out = torch.cat((t_normalized, inp_normalized), dim=1)
        out = self.act_func(self.lin1(out))

        out = self.act_func(out + self.h * self.lin2(out))
        # out = self.act_func(self.lin2(out))
        out = self.act_func(out + self.h * self.lin3(out))
        # out = self.act_func(self.lin3(out))

        out = self.linlast(out)

        ctt = t.view(inp.size(0), 1)
        c1 = ctt / self.TT  # convex weight 1
        c2 = (self.TT - ctt) / self.TT  # convex weight 2
        # 我的
        c3 = 10.0*ctt*(self.TT - ctt)/self.TT
        #return c1 * out + c2 * inp #可恢复的
        return c3*out + c1*samples_from_Terminal_set + c2*inp

