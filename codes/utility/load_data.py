import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

QUERY_NO_YEAR = [58, 90, 96, 106, 132, 134, 163]   # 연도 없는 쿼리 논문은 2020년으로 둔다


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class Data:
    """SPECTER2 임베딩, 논문 연도, 리뷰어 과거 논문, 평가 정답을 읽는다. 행 번호 = 임베딩 파일의 행 순서."""

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        saved = torch.load(self.data_dir / "embedding/specter2_embeddings.pt", map_location="cpu", weights_only=True)
        self.paper_ids = saved["paper_ids"].long().numpy()
        self.embeddings = saved["embeddings"].float()           # [N, 768], 정규화 안 된 SPECTER2
        self.index = pd.Index(self.paper_ids)                   # 논문 ID → 행 번호
        self.n = len(self.paper_ids)

        papers = pd.read_parquet(self.data_dir / "papers_normalized_v2.parquet", columns=["id", "year"])
        self.years = pd.to_numeric(papers.set_index("id")["year"], errors="coerce")   # 원본 연도 (그래프 전처리용)
        self.row_years = self.years.reindex(self.paper_ids).to_numpy(dtype=np.float64)  # 행 순서, 쿼리 7편 2020
        self.row_years[self.index.get_indexer(QUERY_NO_YEAR)] = 2020.0

        # 평가 정답: 폴더의 test_answers_*.jsonl 전부. 순서 soft, hard, soft_25y3, hard_25y3, ...
        def order(f):
            kind, _, suffix = f.stem.replace("test_answers_", "").partition("_")
            return (suffix, 0 if kind == "soft" else 1)
        files = sorted(self.data_dir.glob("test_answers_*.jsonl"), key=order)
        self.answers = {f.stem.replace("test_answers_", ""): read_jsonl(f) for f in files}
        self.query_ids = [int(a["query_id"]) for a in next(iter(self.answers.values()))]

        reviewers = read_jsonl(self.data_dir / "reviewer_papers.jsonl")
        self.reviewers = reviewers
        self.reviewer_ids = sorted((str(r["reviewer_id"]) for r in reviewers), key=int)
        self.history = {}                                       # 리뷰어 → 과거 논문 ID (중복 제거, 임베딩 없는 논문 제외)
        for r in reviewers:
            ids = np.unique(r["paper_ids"])
            self.history[str(r["reviewer_id"])] = ids[self.index.get_indexer(ids) >= 0]

        keep = np.union1d(self.query_ids, np.concatenate(list(self.history.values())))
        self.eval_rows = np.sort(self.index.get_indexer(keep))  # 평가에 필요한 행 (쿼리 + 리뷰어 과거 논문)

    def base_embedding(self, year_mode, prompt_path):
        """MLP·전파에 들어갈 텍스트 임베딩. year_mode 1이면 연도를 넣어 다시 인코딩한 SPECTER2."""
        if year_mode != 1:
            return self.embeddings
        path = Path(prompt_path)
        if not path.is_file():
            raise FileNotFoundError(f"{path} 가 없다. 먼저 python data/build_data.py --prompt 로 만든다.")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        assert np.array_equal(saved["paper_ids"].long().numpy(), self.paper_ids), "prompt 임베딩의 논문 순서가 다르다"
        return saved["embeddings"].float()

    def scaled_year(self):
        """τ = (clip(year, 1970, 2026) − 2008) / 5, 연도 없으면 0. 반환: (τ, 연도 있음 마스크) — 행 순서 텐서."""
        valid = ~np.isnan(self.row_years)
        tau = np.nan_to_num((np.clip(self.row_years, 1970, 2026) - 2008) / 5, nan=0.0)
        return torch.as_tensor(tau, dtype=torch.float32), torch.as_tensor(valid, dtype=torch.float32)
