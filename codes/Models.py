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


def propagate(adj, x, n_layers, rows=slice(None)):
    """mean(x, Âx, ..., Â^L x)[rows] (역전파 가능). rows 행만 더해 두어 메모리를 아낀다."""
    acc, h = x[rows], x
    for _ in range(n_layers):
        h = SpMM.apply(adj, h)
        acc = acc + h[rows]
    return acc / (n_layers + 1)


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


class ProjectionHead(nn.Module):
    """GRACE의 2층 projection (두 view 공유). loss에만 쓰고, 평가는 projection 전 임베딩으로 한다."""

    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.ELU(), nn.Linear(dim, dim))

    def forward(self, z):
        return self.net(z)


def grace_loss(h1, h2, tau=0.5):
    """GRACE NT-Xent. positive = 같은 노드의 두 view, negative = batch 안 다른 노드 (view 사이 + view 안)."""
    h1, h2 = F.normalize(h1, dim=1), F.normalize(h2, dim=1)
    e12, e11, e22 = (torch.exp(a @ b.T / tau) for a, b in ((h1, h2), (h1, h1), (h2, h2)))
    pos = e12.diag()
    l1 = -torch.log(pos / (e12.sum(1) + e11.sum(1) - e11.diag()))
    l2 = -torch.log(pos / (e12.sum(0) + e22.sum(1) - e22.diag()))
    return 0.5 * (l1 + l2).mean()


class LightGCNGCL(nn.Module):
    """kNN 그래프 view × 인용 그래프 view 대조학습 (GRACE) 후 LightGCN 전파.

      view1: T1 = E0 (고정)          ── kNN 그래프 Â_knn ── LightGCN ──▶ z1
      view2: T2 = E0로 시작 (학습)    ── view2 그래프 Â_v2 ── LightGCN ──▶ z2
      L = GRACE( proj(z1), proj(z2) ),  proj = 2층 projection (두 view 공유)
      최종 H = 학습된 T2를 input graph Â로 전파

    E0 = [연도 특징 | 텍스트 임베딩] (LightGCN과 같은 year_mode).
      year_mode 0·1: T2 = 텍스트 임베딩 / 2: T2 = [τ | SPECTER2] (τ 열도 학습)
      year_mode 3: T2 = SPECTER2, Time2Vec(ω, φ)은 두 view가 공유해 학습
    """

    def __init__(self, x, tau, tau_valid, adj, adj_knn, adj_v2, year_mode=0, n_layers=2, t2v_dim=17, seed=0):
        """adj_knn은 고정된 view1을 한 번 전파하는 데만 쓰고 들고 있지 않는다."""
        super().__init__()
        self.device = device = adj.device
        self.adj, self.adj_v2, self.year_mode, self.n_layers = adj, adj_v2, year_mode, n_layers
        x = x.to(device)
        if year_mode == 2:
            x = torch.cat([tau.to(device).unsqueeze(1), x], dim=1)
        self.tau, self.tau_valid = tau.to(device), tau_valid.to(device)
        self.t2v = Time2Vec(t2v_dim, seed).to(device) if year_mode == 3 else None
        self.dim = x.shape[1] + (t2v_dim if year_mode == 3 else 0)
        with torch.no_grad():
            self.z1 = propagate(adj_knn, x, n_layers)                 # T1은 고정이라 한 번만 전파
            if year_mode == 3:
                # Time2Vec 블록은 연도에만 의존 → 연도 one-hot을 kNN으로 미리 전파해 두면 z1 블록 = M · T2V(연도값)
                key = torch.where(self.tau_valid > 0, self.tau, torch.full_like(self.tau, -100.0))
                uniq, inv = torch.unique(key, return_inverse=True)
                self.uniq_valid = (uniq != -100.0).float()
                self.uniq_tau = uniq * self.uniq_valid
                onehot = torch.zeros(len(key), len(uniq), device=device)
                onehot[torch.arange(len(key), device=device), inv] = 1.0
                self.m_knn = propagate(adj_knn, onehot, n_layers)
                del onehot
        self.t2 = nn.Parameter(x)

    def year_block(self):
        return self.t2v(self.tau, self.tau_valid)

    def view1(self, rows):
        z = self.z1[rows]
        if self.t2v is not None:
            z = torch.cat([self.m_knn[rows] @ self.t2v(self.uniq_tau, self.uniq_valid), z], dim=1)
        return z

    def view2(self, rows, adj=None):
        adj = self.adj_v2 if adj is None else adj
        z = propagate(adj, self.t2, self.n_layers, rows)
        if self.t2v is not None:
            z = torch.cat([propagate(adj, self.year_block(), self.n_layers, rows), z], dim=1)
        return z

    def loss(self, rows, proj, tau):
        return grace_loss(proj(self.view1(rows)), proj(self.view2(rows)), tau)

    @torch.no_grad()
    def output(self, rows):
        """rows 행의 최종 임베딩 H[rows] = 학습된 T2를 input graph로 전파 (CPU)."""
        return self.view2(torch.as_tensor(rows, device=self.device), self.adj).cpu()
