"""전처리한 인용 그래프 + LightGCN (+MLP, +GCL, +연도) 리뷰어 추천. 사용법은 README.md.

※ 처음 실행하기 전에 임베딩부터 만든다 (한 번만, GPU 수 시간):
  1) dataset/에 데이터 파일을 넣는다 (README "데이터 준비")
  2) python data/build_data.py --specter2     # → dataset/embedding/specter2_embeddings.pt (필수)
  3) python data/build_data.py --prompt       # → cache/prompt_year_embeddings.pt (--year_mode 1을 쓸 때만)
  그래프·valid·kNN 그래프는 main.py가 처음 실행할 때 자동으로 만든다.

  python main.py --model specter2 --topk 0 --review_year 0
  python main.py --model lightgcn --sim_percent 100 --year_gap none --mlp 0 --year_mode 0 --topk 1
  python main.py --model lightgcn --sim_percent 75 --year_gap 3 --mlp 1 --year_mode 3 --topk 0 --review_year 3
  python main.py --model lightgcn+gcl --knn_k 100 --view2_sim_percent 50 --view2_year_gap 5 --year_mode 3 --topk 0 --review_year 0
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from Models import LightGCN, LightGCNGCL, ProjectionHead, norm_adj
from data.build_data import build_graph, build_knn, build_valid, query_citing_pairs
from utility.batch_test import test, ValidScorer
from utility.load_data import Data
from utility.metrics import METRICS
from utility.parser import parse_args


def log(msg, t0):
    print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)


def save_results(out, args, results, extra=None):
    """results: {(k, n): (요약, 쿼리별)}. metrics.csv = 조합 × 세트 × 지표 10개 (긴 형식)."""
    rows, pers = [], []
    for (k, n), (summary, per) in results.items():
        for ds, m in summary.items():
            rows.append({"topk": k, "review_year": n if n is not None else "none", "dataset": ds, **m})
        pers.append(per.assign(topk=k, review_year=n if n is not None else "none"))
    table = pd.DataFrame(rows)[["topk", "review_year", "dataset", "queries"] + METRICS]
    table.to_csv(out / "metrics.csv", index=False)
    pd.concat(pers).to_csv(out / "per_query.csv", index=False)
    (out / "metrics.json").write_text(json.dumps({"config": vars(args), **(extra or {})}, indent=2, ensure_ascii=False))
    for metric in ("P@5", "P@10"):
        wide = table.pivot_table(index=["topk", "review_year"], columns="dataset", values=metric, sort=False)
        print(f"\n{metric}\n" + wide[list(results[next(iter(results))][0])].round(4).to_string())


class NegSampler:
    """anchor마다 train 노드 중 무작위 1개. 자기 자신이거나 anchor와 인용 edge가 있으면 다시 뽑는다(최대 20번, 남으면 loss에서 뺀다)."""

    def __init__(self, nodes, edge_pairs, n):
        self.nodes, self.n = nodes, n
        self.keys = torch.sort(edge_pairs[:, 0] * n + edge_pairs[:, 1]).values

    def is_edge(self, a, b):
        k = torch.minimum(a, b) * self.n + torch.maximum(a, b)
        pos = torch.searchsorted(self.keys, k).clamp_max(len(self.keys) - 1)
        return self.keys[pos] == k

    def sample(self, a):
        draw = lambda m: self.nodes[torch.randint(len(self.nodes), (m,), device=a.device)]
        ng = draw(len(a))
        for _ in range(20):
            bad = (ng == a) | self.is_edge(a, ng)
            if not bad.any():
                break
            ng[bad] = draw(int(bad.sum()))
        return ng, (ng != a) & ~self.is_edge(a, ng)


def train(args, data, model, device, out, t0):
    """MLP 학습. positive = train graph edge, negative = 무작위, BPR(cos/τ). valid MRR 최고 에폭의 체크포인트를 돌려준다."""
    valid = build_valid(data, args.cache_dir)
    pairs = build_graph(data, args.train_sim_percent, args.train_year_gap, args.cache_dir, device)
    later = query_citing_pairs(data)                                   # 누수 방지: 가짜 쿼리, 쿼리를 인용한 edge 제외
    bad = np.isin(pairs, valid["pseudo_rows"]).any(1) | np.array([(a, b) in later for a, b in map(tuple, pairs)])
    pairs = torch.as_tensor(pairs[~bad], device=device)
    full = torch.as_tensor(build_graph(data, 100, None, args.cache_dir, device), device=device)
    sampler = NegSampler(torch.unique(pairs), full, data.n)
    del full
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=args.lr)
    scorer = ValidScorer(valid)
    P, B = len(pairs), args.batch
    steps = math.ceil(P / B)
    log(f"train positive {P} (제외 {int(bad.sum())}), 에폭당 {steps} step, 학습 파라미터 {sum(p.numel() for p in params)}", t0)

    curve, best = [], {"MRR": -1.0, "epoch": -1, "state": None}

    def checkpoint(ep, loss):
        m = scorer(model.output(scorer.need_rows))
        torch.cuda.empty_cache()
        curve.append({"epoch": ep, "loss": loss, "valid_MRR": m["MRR"], "valid_P@5": m["P@5"], "time_s": round(time.time() - t0)})
        pd.DataFrame(curve).to_csv(out / "curve.csv", index=False)
        improved = m["MRR"] > best["MRR"]
        if improved:
            best.update(MRR=m["MRR"], epoch=ep, state=copy.deepcopy(model.state_dict()))
        log(f"epoch {ep:3d} | loss {loss if loss is not None else float('nan'):.4f} | valid MRR {m['MRR']:.4f} "
            f"(best {best['MRR']:.4f} @ {best['epoch']})", t0)
        return improved

    checkpoint(0, None)
    no_improve, stop = 0, args.max_epochs
    for ep in range(1, args.max_epochs + 1):
        perm = torch.randperm(P, device=device)
        total = 0.0
        for s in range(steps):
            i = perm[s * B:(s + 1) * B]
            sw = torch.rand(len(i), device=device) < 0.5               # anchor = edge 양 끝 중 무작위
            a, b = torch.where(sw, pairs[i, 1], pairs[i, 0]), torch.where(sw, pairs[i, 0], pairs[i, 1])
            ng, ok = sampler.sample(a)
            H = model()
            za, zb, zn = (F.normalize(H[x], dim=1) for x in (a, b, ng))
            diff = ((za * zb).sum(1) - (za * zn).sum(1)) / args.tau
            loss = (F.softplus(-diff) * ok).sum() / ok.sum().clamp_min(1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item()
        if ep % args.eval_every == 0:
            if checkpoint(ep, total / steps):
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= args.patience:
                    stop = ep
                    log(f"얼리스타핑: valid {args.patience}번 연속 개선 없음 → epoch {ep}", t0)
                    break
        else:
            log(f"epoch {ep:3d} | loss {total / steps:.4f}", t0)
    model.load_state_dict(best["state"])
    return {"best_epoch": best["epoch"], "stop_epoch": stop, "best_valid_MRR": best["MRR"], "valid_MRR_epoch0": curve[0]["valid_MRR"]}


def train_gcl(args, data, model, device, out, t0):
    """대조학습 (GRACE). step마다 연도 있는 논문 중 무작위 gcl_batch개의 두 view를 맞춘다.
    정해진 step만큼 돌고 마지막 모델로 평가한다 (valid·얼리스타핑 없음)."""
    proj = ProjectionHead(model.dim).to(device)
    params = list(model.parameters()) + list(proj.parameters())
    opt = torch.optim.Adam(params, lr=args.gcl_lr)
    nodes = torch.as_tensor(np.flatnonzero(~np.isnan(data.row_years)), device=device)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    t2_init = model.t2.detach().cpu().clone()
    log(f"GCL 학습 노드 {len(nodes)}, step {args.gcl_steps} × batch {args.gcl_batch}, "
        f"학습 파라미터 {sum(p.numel() for p in params)}", t0)
    curve = []
    for step in range(1, args.gcl_steps + 1):
        rows = nodes[torch.randint(len(nodes), (args.gcl_batch,), device=device, generator=gen)]
        loss = model.loss(rows, proj, args.gcl_tau)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        curve.append({"step": step, "loss": loss.item(), "time_s": round(time.time() - t0)})
        if step == 1 or step % 25 == 0 or step == args.gcl_steps:
            log(f"step {step:3d} | loss {curve[-1]['loss']:.4f}", t0)
    pd.DataFrame(curve).to_csv(out / "curve.csv", index=False)
    del opt, proj, params
    model.t2.grad = None
    torch.cuda.empty_cache()
    t2_cos = float(F.cosine_similarity(model.t2.detach().cpu(), t2_init, dim=1).median())
    log(f"학습 후 T2 vs E0 cos 중앙값 {t2_cos:.4f}", t0)
    return {"loss_step1": curve[0]["loss"], "loss_last": curve[-1]["loss"], "T2_vs_E0_cos_median": t2_cos}


def main():
    args = parse_args()
    emb = Path(args.data_dir) / "embedding/specter2_embeddings.pt"
    if not emb.is_file():
        raise SystemExit(f"{emb} 가 없다. 먼저 SPECTER2 임베딩을 만든다: python data/build_data.py --specter2 "
                         f"(데이터 폴더가 다르면 --data_dir도 같이). README '데이터 준비' 참고.")
    if args.year_mode == 1 and not Path(args.prompt_path).is_file():
        raise SystemExit(f"{args.prompt_path} 가 없다. --year_mode 1은 먼저 python data/build_data.py --prompt 를 실행한다.")
    t0 = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    out = Path(args.save_dir) / args.model_name
    out.mkdir(parents=True, exist_ok=True)
    log(f"{args.model_name} → {out}", t0)

    data = Data(args.data_dir)
    x = data.base_embedding(args.year_mode, args.prompt_path)
    tau, tau_valid = data.scaled_year()
    if args.model == "specter2":                                       # 전파 없음: H = E0
        adj, n_layers = None, 0
    else:
        pairs = build_graph(data, args.sim_percent, args.year_gap, args.cache_dir, device)
        adj, n_layers = norm_adj(pairs, data.n, device), args.n_layers
        log(f"input graph s{args.sim_percent}_y{args.year_gap}: 무방향 edge {len(pairs)}", t0)
    if args.model == "lightgcn+gcl":
        knn = build_knn(data, args.knn_k, args.cache_dir, device)
        adj_knn = norm_adj(knn, data.n, device)
        v2 = build_graph(data, args.view2_sim_percent, args.view2_year_gap, args.cache_dir, device)
        log(f"view1 kNN k{args.knn_k}: 무방향 edge {len(knn)} / view2 s{args.view2_sim_percent}_y{args.view2_year_gap}: "
            f"무방향 edge {len(v2)}", t0)
        model = LightGCNGCL(x, tau, tau_valid, adj, adj_knn, norm_adj(v2, data.n, device), args.year_mode, n_layers,
                            args.t2v_dim, args.seed)
        del x, knn, adj_knn, v2
        torch.cuda.empty_cache()
        extra = train_gcl(args, data, model, device, out, t0)
    else:
        model = LightGCN(x, tau, tau_valid, adj, args.year_mode, args.mlp, args.hidden, args.dim, n_layers,
                         args.t2v_dim, args.seed, device)
        del x
        extra = train(args, data, model, device, out, t0) if args.mlp else {}
    H = model.output(data.eval_rows)
    results = test(data, data.eval_rows, H, args.topks, args.review_years)
    save_results(out, args, results, extra)
    log("완료", t0)


if __name__ == "__main__":
    main()
