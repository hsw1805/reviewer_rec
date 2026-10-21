import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def norm_adj(pairs, n, device):
    """무방향 쌍 [U, 2] → Â = D^-1/2 A D^-1/2 (대칭, sparse CSR)."""
    p = torch.as_tensor(pairs, device=device)
    row, col = torch.cat([p[:, 0], p[:, 1]]), torch.cat([p[:, 1], p[:, 0]])
    inv = torch.bincount(row, minlength=n).float().clamp_min(1).rsqrt()
    return torch.sparse_coo_tensor(torch.stack([row, col]), inv[row] * inv[col], (n, n)).coalesce().to_sparse_csr()


class SpMM(torch.autograd.Function):
    """y = Â x. Â가 대칭이라 역전파도 Â g."""

    @staticmethod
    def forward(ctx, adj, x):
        ctx.adj = adj
        return torch.sparse.mm(adj, x)

    @staticmethod
    def backward(ctx, g):
        return None, torch.sparse.mm(ctx.adj, g.contiguous())


class Time2Vec(nn.Module):
    """[ω0·τ + φ0, sin(ω_i·τ + φ_i)]_{i=1..k-1}. 연도 없는 논문은 0벡터. ω ~ N(0, 0.5²), φ ~ U(0, 2π) (seed 고정)."""

    def __init__(self, k=17, seed=0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.w = nn.Parameter(torch.randn(k, generator=gen) * 0.5)
        self.b = nn.Parameter(torch.rand(k, generator=gen) * 2 * math.pi)

    def forward(self, tau, valid):
        z = tau.unsqueeze(1) * self.w + self.b
        return torch.cat([z[:, :1], torch.sin(z[:, 1:])], dim=1) * valid.unsqueeze(1)


class LightGCN(nn.Module):
    """초기 임베딩 E0 → H = (E0 + ÂE0 + ... + Â^L E0) / (L + 1).

    E0 (year_mode, mlp에 따라):
      입력 특징 X = [연도 특징 | 텍스트 임베딩]
        year_mode 0: 연도 특징 없음, 텍스트 = SPECTER2
        year_mode 1: 연도 특징 없음, 텍스트 = 연도를 넣어 다시 인코딩한 SPECTER2 (prompt year)
        year_mode 2: 연도 특징 = τ (1차원), τ = (year − 2008) / 5
        year_mode 3: 연도 특징 = Time2Vec(τ) (t2v_dim차원)
      mlp 0: E0 = X (학습 없음)
      mlp 1: E0 = normalize(Linear(hidden→dim)(ReLU(Linear(X→hidden)))) — 텍스트 임베딩은 고정, MLP(+Time2Vec)만 학습
    """

    def __init__(self, x, tau, tau_valid, adj, year_mode=0, mlp=0, hidden=256, dim=64, n_layers=2, t2v_dim=17, seed=0,
                 device=None):
        """adj=None, n_layers=0이면 전파 없이 E0 그대로 (= SPECTER2 모델)."""
        super().__init__()
        self.adj, self.year_mode, self.use_mlp, self.n_layers = adj, year_mode, mlp, n_layers
        device = device if adj is None else adj.device
        self.device = device
        self.x = x.to(device) if mlp else x                         # mlp 0이면 CPU에 두고 열 블록씩 GPU로
        self.tau, self.tau_valid = tau.to(device), tau_valid.to(device)
        self.t2v = Time2Vec(t2v_dim, seed) if year_mode == 3 else None
        year_dim = {2: 1, 3: t2v_dim}.get(year_mode, 0)
        if mlp:
            # Linear([연도 | 텍스트]) = Linear_x(텍스트) + Linear_y(연도): 89만 × 785 concat을 만들지 않으려고 나눠 둔다
            self.lin_x = nn.Linear(x.shape[1], hidden)
            self.lin_y = nn.Linear(year_dim, hidden, bias=False) if year_dim else None
            self.lin_out = nn.Linear(hidden, dim)
        self.to(device)

    def year_feature(self):
        if self.year_mode == 2:
            return self.tau.unsqueeze(1)
        if self.year_mode == 3:
            return self.t2v(self.tau, self.tau_valid)
        return None

    def forward(self):
        """mlp 1: 전체 노드의 H [N, dim] (역전파 가능)."""
        z = self.lin_x(self.x)
        y = self.year_feature()
        if y is not None:
            z = z + self.lin_y(y)
        e = F.normalize(self.lin_out(F.relu(z)), dim=1)
        acc = e
        for _ in range(self.n_layers):
            e = SpMM.apply(self.adj, e)
            acc = acc + e
        return acc / (self.n_layers + 1)

    @torch.no_grad()
    def output(self, rows, block=64):
        """rows 행의 최종 임베딩 H[rows] (CPU)."""
        rows_t = torch.as_tensor(rows, device=self.device)
        if self.use_mlp:
            return self.forward()[rows_t].cpu()
        y = self.year_feature()
        cols = ([y.cpu()] if y is not None else []) + [self.x[:, i:i + block] for i in range(0, self.x.shape[1], block)]
        out = []
        for c in cols:                                               # 열 블록마다 전파 (메모리 절약)
            e = c.to(self.device).contiguous()
            acc = e.clone()
            for _ in range(self.n_layers):
                e = torch.sparse.mm(self.adj, e)
                acc += e
            out.append((acc / (self.n_layers + 1))[rows_t].cpu())
        return torch.cat(out, dim=1)
