import torch
from torch.autograd import grad
import torch.nn.functional as F

class ReintegrationTracking:

    def __init__(self, SX=256, SY=256, dt=.2, dd=5, sigma=.65, has_hidden=False, mix="softmax"):
        self.SX = SX
        self.SY = SY
        self.dt = dt
        self.dd = dd
        self.sigma = sigma
        self.has_hidden = has_hidden
        self.mix = mix

        self.apply = self._build_apply()

    def __call__(self, *args):
        return self.apply(*args)

    def _build_apply(self):

        x, y = torch.arange(self.SX), torch.arange(self.SY)
        X, Y = torch.meshgrid(x, y)
        pos = torch.stack((X, Y), dim=-1) + .5  # (SX, SY, 2)
        dxs = []
        dys = []
        dd = self.dd
        for dx in range(-dd, dd+1):
            for dy in range(-dd, dd+1):
                dxs.append(dx)
                dys.append(dy)
        dxs = torch.tensor(dxs)
        dys = torch.tensor(dys)

        def step(X, mu, dx, dy):
            Xr = torch.roll(X, (dx, dy), dims=(0, 1))
            mur = torch.roll(mu, (dx, dy), dims=(0, 1))
            dpmu = torch.min(torch.stack(
                    [torch.abs(pos[..., None] - (mur + torch.tensor([di, dj])[None, None, :, None]))
                    for di in (-self.SX, 0, self.SX) for dj in (-self.SY, 0, self.SY)]
                ), dim=0)[0]

#            dpmu = torch.absolute(pos[..., None] - mur)


            sz = 0.5 - dpmu + self.sigma
            area = torch.prod(torch.clamp(sz, 0, min(1, 2*self.sigma)) , dim=2) / (4 * self.sigma**2)
            nX = Xr * area

            return nX


        def apply(X, F):
            ma = self.dd - self.sigma  # upper bound of the flow magnitude
            mu = pos[..., None] + torch.clamp(self.dt * F, -ma, ma)  # (x, y, 2, c) : target positions (distribution centers)
            
      #      mu = torch.clamp(mu, self.sigma, self.SX-self.sigma)

            nX = torch.zeros_like(X)  # initialize nX

            for dx, dy in zip(dxs, dys):
                nX += step(X, mu, dx, dy)

            return nX
        return apply
