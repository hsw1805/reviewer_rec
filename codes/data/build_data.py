"""데이터 전처리: input graph, valid 세트, prompt year 임베딩. 결과는 cache_dir에 저장하고 main.py가 다시 쓴다.

  python data/build_data.py --sim_percent 75 --year_gap 3     # 그래프 하나 미리 만들기
  python data/build_data.py --valid                           # valid 세트
  python data/build_data.py --prompt                          # year_mode 1용 임베딩 (89만 편 다시 인코딩, GPU 수 시간)
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utility.load_data import Data, QUERY_NO_YEAR  # noqa: E402

N_PSEUDO, N_CAND, VALID_SEED = 3000, 20, 42


# ---------------------------------------------------------------------- input graph
def build_graph(data, sim_percent, year_gap, cache_dir, device="cpu"):
    """인용 그래프 전처리. 반환: 무방향 쌍 [U, 2] (행 번호, a < b).
    ① 중복·자기 인용 제거, 양 끝 임베딩·연도 있는 edge만 ② source별 SPECTER2 cos 상위 ceil(n·sim%) ③ |연도 차| ≤ gap
    (연도는 원본 그대로 — 연도 없는 쿼리 7편의 edge는 ①에서 빠진다. 예전 실험과 같다.)"""
    path = Path(cache_dir) / f"graph_s{sim_percent}_y{year_gap}.npy"
    if path.is_file():
        return np.load(path)
    s, t = "source_paper_id", "cited_paper_id"
    g = pd.read_parquet(data.data_dir / "edges.parquet", columns=[s, t]).drop_duplicates()
    g = g[g[s].isin(data.index) & g[t].isin(data.index) & g[s].ne(g[t])]
    g = g[g[s].map(data.years).notna() & g[t].map(data.years).notna()].copy()
    if sim_percent < 100:
        src, dst = data.index.get_indexer(g[s]), data.index.get_indexer(g[t])
        sims = np.empty(len(g), dtype=np.float32)
        with torch.inference_mode():
            for i in range(0, len(g), 8192):
                a = data.embeddings[torch.as_tensor(src[i:i + 8192])].to(device)
                b = data.embeddings[torch.as_tensor(dst[i:i + 8192])].to(device)
                sims[i:i + 8192] = F.cosine_similarity(a, b, dim=1).cpu().numpy()
        g["similarity"] = sims
        g = g.sort_values([s, "similarity", t], ascending=[True, False, True])
        grp = g.groupby(s, sort=False)
        g = g[grp.cumcount() + 1 <= np.ceil(grp[t].transform("size") * sim_percent / 100)]
    if year_gap is not None:
        g = g[(g[s].map(data.years) - g[t].map(data.years)).abs() <= year_gap]
    a, b = data.index.get_indexer(g[s]), data.index.get_indexer(g[t])
    pairs = np.unique(np.column_stack([np.minimum(a, b), np.maximum(a, b)]), axis=0).astype(np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, pairs)
    return pairs


def query_citing_pairs(data):
    """쿼리 논문을 인용한 edge (쿼리보다 나중 논문 → 학습에 쓰면 누수). 무방향 쌍 집합 (행 번호)."""
    e = pd.read_parquet(data.data_dir / "edges.parquet").query("cited_paper_id <= 174")
    a, b = data.index.get_indexer(e.source_paper_id), data.index.get_indexer(e.cited_paper_id)
    return {(min(x, y), max(x, y)) for x, y in zip(a, b)}


# ---------------------------------------------------------------------- valid
def build_valid(data, cache_dir):
    """가짜 쿼리 = 리뷰어 과거 논문 중 2018~2020년, 참고문헌 ≥5, 저자 리뷰어가 모두 이력 ≥2편, 저자 ≤10명인 논문 3,000편 (seed 42).
    후보 = 저자 리뷰어 + 나머지 무작위, 총 20명. 평가 정답은 쓰지 않는다."""
    path = Path(cache_dir) / "valid_set.npz"
    if path.is_file():
        z = np.load(path, allow_pickle=True)
        return {k: z[k] for k in z.files}
    e = pd.read_parquet(data.data_dir / "edges.parquet").drop_duplicates()
    e = e[e.source_paper_id != e.cited_paper_id]
    size = int(pd.read_parquet(data.data_dir / "papers.parquet", columns=["id"])["id"].max()) + 1
    outdeg = np.bincount(e.source_paper_id.to_numpy(), minlength=size)
    history = {str(r["reviewer_id"]): np.unique(r["paper_ids"]) for r in data.reviewers}   # 임베딩 없는 논문 포함
    H = np.unique(np.concatenate(list(history.values())))
    owners = {}
    for rid, papers in history.items():
        for p in papers:
            owners.setdefault(int(p), []).append(rid)
    hist_len = {rid: len(v) for rid, v in history.items()}
    pool = [p for p in H
            if p > 174 and 2018 <= data.years.get(p, np.nan) <= 2020 and outdeg[p] >= 5
            and all(hist_len[r] >= 2 for r in owners[int(p)]) and len(owners[int(p)]) <= N_CAND // 2]
    rng = np.random.default_rng(VALID_SEED)
    pseudo = np.sort(rng.choice(pool, size=N_PSEUDO, replace=False))
    rid_pos = {rid: j for j, rid in enumerate(data.reviewer_ids)}
    cands, labels = [], []
    for p in pseudo:
        pos = sorted(owners[int(p)], key=int)
        others = [r for r in data.reviewer_ids if r not in set(pos)]
        neg = list(rng.choice(others, size=N_CAND - len(pos), replace=False))
        cand = sorted(pos + neg, key=int)
        cands.append([rid_pos[r] for r in cand])
        labels.append([int(r in pos) for r in cand])
    h_pos = pd.Index(H)
    hist_cols = np.empty(len(data.reviewer_ids), dtype=object)
    for j, rid in enumerate(data.reviewer_ids):
        hist_cols[j] = h_pos.get_indexer(history[rid])
    out = {"pseudo_ids": pseudo, "pseudo_rows": data.index.get_indexer(pseudo), "hide_cols": h_pos.get_indexer(pseudo),
           "H_ids": H, "H_rows": data.index.get_indexer(H), "hist_cols": hist_cols,
           "cands": np.array(cands), "labels": np.array(labels)}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)
    return out


# ---------------------------------------------------------------------- prompt year 임베딩
def build_prompt_embeddings(data_dir, cache_dir, chunk=50000, tokens_per_batch=16384):
    """SPECTER2(specter2_base + allenai/specter2 adapter) 입력에 연도를 넣어 다시 인코딩.
    입력: title + [SEP] + "Year: {year}. " + abstract (연도 없으면 "Year: unknown.", 쿼리 7편은 2020). CLS, max_len 512, fp16."""
    from adapters import AutoAdapterModel
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("allenai/specter2_base")
    model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
    model.load_adapter("allenai/specter2", source="hf", load_as="specter2", set_active=True)
    model = model.to("cuda").eval()

    papers = pd.read_parquet(Path(data_dir) / "papers_normalized_v2.parquet", columns=["id", "title", "abstract", "year"])
    years = pd.to_numeric(papers["year"], errors="coerce").to_numpy()
    years[np.isin(papers["id"].to_numpy(), QUERY_NO_YEAR)] = 2020
    texts = [t + tok.sep_token + f"Year: {'unknown' if np.isnan(y) else int(y)}." + (" " + a if a else "")
             for t, a, y in zip(papers["title"].fillna("").astype(str), papers["abstract"].fillna("").astype(str), years)]

    @torch.inference_mode()
    def encode(batch_texts):
        enc = tok(batch_texts, truncation=True, max_length=512, return_token_type_ids=False)["input_ids"]
        order = np.argsort([len(x) for x in enc])                      # 길이순 배치
        out, i = torch.empty(len(batch_texts), 768), 0
        while i < len(order):
            j = i
            while j < len(order) and (j - i + 1) * len(enc[order[j]]) <= tokens_per_batch:
                j += 1
            idx = order[i:max(j, i + 1)]
            b = tok.pad({"input_ids": [enc[k] for k in idx]}, return_tensors="pt").to("cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                out[torch.as_tensor(idx)] = model(**b).last_hidden_state[:, 0, :].float().cpu()
            i = max(j, i + 1)
        return out

    tmp = Path(cache_dir) / "prompt_year_chunks"                       # 끊겨도 이어서 돈다
    tmp.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_chunks = (len(texts) + chunk - 1) // chunk
    for c in range(n_chunks):
        p = tmp / f"chunk_{c:03d}.pt"
        if not p.is_file():
            torch.save(encode(texts[c * chunk:(c + 1) * chunk]), p)
            print(f"chunk {c + 1}/{n_chunks} {time.time() - t0:.0f}s", flush=True)
    emb = torch.cat([torch.load(tmp / f"chunk_{c:03d}.pt") for c in range(n_chunks)])
    torch.save({"paper_ids": torch.as_tensor(papers["id"].to_numpy()), "embeddings": emb},
               Path(cache_dir) / "prompt_year_embeddings.pt")


if __name__ == "__main__":
    from utility.parser import parse_args
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--valid", action="store_true")
    ap.add_argument("--prompt", action="store_true")
    extra, rest = ap.parse_known_args()
    args = parse_args(rest)
    if extra.prompt:
        build_prompt_embeddings(args.data_dir, args.cache_dir)
    else:
        data = Data(args.data_dir)
        if extra.valid:
            v = build_valid(data, args.cache_dir)
            print("valid 가짜 쿼리", len(v["pseudo_ids"]))
        else:
            dev = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
            pairs = build_graph(data, args.sim_percent, args.year_gap, args.cache_dir, dev)
            print(f"graph s{args.sim_percent}_y{args.year_gap}: 무방향 edge {len(pairs)} (방향 {2 * len(pairs)})")
