import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from utility.metrics import compute_ranking_metrics, evaluate_reviewer_scores


REF_YEAR, EMPTY_SCORE = 2020, -10.0


def reviewer_scores(data, rows, H, topks=(1,), review_years=(None,)):
    """리뷰어 점수 표 {(k, n): DataFrame(행 query_id, 열 reviewer_id)}. rows: H의 각 행이 가리키는 임베딩 행 번호.
    ① 연도 n이 있으면 리뷰어 과거 논문 중 연도 ≥ 2020 − n인 논문만 남긴다 (연도 없는 논문은 뺀다).
    ② 남은 과거 논문 중 쿼리와 cos 상위 k편의 평균 (k보다 적게 남으면 남은 만큼 평균).
    남은 논문이 없는 리뷰어는 최하위 점수(−10)."""
    pos = pd.Index(rows)
    Hn = F.normalize(H.float(), dim=1)
    q = Hn[pos.get_indexer(data.index.get_indexer(data.query_ids))]
    out = {(k, n): {} for k in topks for n in review_years}
    for r in data.reviewer_ids:
        ids = data.history[r]
        cos = (q @ Hn[pos.get_indexer(data.index.get_indexer(ids))].T).numpy()   # [쿼리, 과거 논문]
        py = data.years.reindex(ids).to_numpy()
        for n in review_years:
            c = cos if n is None else cos[:, py >= REF_YEAR - n]
            srt = -np.sort(-c, axis=1)
            for k in topks:
                out[(k, n)][r] = srt[:, :k].mean(1) if c.shape[1] else np.full(len(q), EMPTY_SCORE)
    return {key: pd.DataFrame(v, index=data.query_ids)[data.reviewer_ids] for key, v in out.items()}


def test(data, rows, H, topks=(1,), review_years=(None,)):
    """조합 (k, n)마다 평가 정답 전부 × 지표 10개. 반환: {(k, n): (요약 {세트: {지표: 값}}, 쿼리별 DataFrame)}."""
    return {key: evaluate_reviewer_scores(s, data.answers)
            for key, s in reviewer_scores(data, rows, H, topks, review_years).items()}


class ValidScorer:
    """valid: 리뷰어 과거 논문 3,000편을 가짜 쿼리로, 그 논문을 숨긴 이력으로 max cos → 후보 20명 중 저자 맞히기.
    평가 정답은 쓰지 않는다. 체크포인트(에폭) 선택에만 쓴다."""

    def __init__(self, valid):
        self.v = valid
        self.h_valid = valid["H_rows"] >= 0
        self.need_rows = np.unique(np.concatenate([valid["pseudo_rows"], valid["H_rows"][self.h_valid]]))
        self.pos = pd.Index(self.need_rows)
        self.hist = [torch.as_tensor(c) for c in valid["hist_cols"]]

    @torch.no_grad()
    def __call__(self, H_need):
        """H_need: need_rows 순서의 최종 임베딩 (CPU)."""
        x = F.normalize(H_need.float(), dim=1)
        q = x[torch.as_tensor(self.pos.get_indexer(self.v["pseudo_rows"]))]
        h = torch.zeros(len(self.v["H_rows"]), x.shape[1])
        hv = np.flatnonzero(self.h_valid)
        h[torch.as_tensor(hv)] = x[torch.as_tensor(self.pos.get_indexer(self.v["H_rows"][hv]))]
        cos = q @ h.T
        cos[:, torch.as_tensor(~self.h_valid)] = -float("inf")
        cos[torch.arange(len(q)), torch.as_tensor(self.v["hide_cols"])] = -float("inf")
        cos_r = torch.stack([cos[:, c].max(dim=1).values for c in self.hist], dim=1).numpy()
        res = [compute_ranking_metrics(l, cos_r[i, c]) for i, (c, l) in enumerate(zip(self.v["cands"], self.v["labels"]))]
        return pd.DataFrame(res).mean().to_dict()
