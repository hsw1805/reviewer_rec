import numpy as np
import pandas as pd

METRICS = ["P@1", "P@2", "P@3", "P@4", "P@5", "P@10", "NDCG@5", "NDCG@10", "MRR", "AUC"]


def compute_ranking_metrics(labels, scores):
    """한 쿼리의 후보 리뷰어 지표. 동점은 입력 순서(리뷰어 ID 순). P@k 분모는 항상 k.
    AUC 동점 0.5, 정답이 없거나 전부 정답이면 NaN (평균에서 빠짐)."""
    labels = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    ranked = labels[np.argsort(-scores, kind="stable")]
    positives = int(labels.sum())
    m = {f"P@{k}": float(ranked[:k].sum() / k) for k in (1, 2, 3, 4, 5, 10)}
    for k in (5, 10):
        disc = 1.0 / np.log2(np.arange(2, min(k, len(ranked)) + 2))
        ideal = float(disc[:min(positives, k)].sum())
        m[f"NDCG@{k}"] = float((ranked[:k] * disc).sum()) / ideal if ideal > 0 else 0.0
    hits = np.flatnonzero(ranked == 1)
    m["MRR"] = float(1.0 / (hits[0] + 1)) if len(hits) else 0.0
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) and len(neg):
        diff = pos[:, None] - neg[None, :]
        m["AUC"] = float(((diff > 0) + 0.5 * (diff == 0)).mean())
    else:
        m["AUC"] = float("nan")
    return m


def evaluate_reviewer_scores(scores, answers):
    """scores: DataFrame(행 query_id, 열 reviewer_id 문자열). answers: {세트 이름: 정답 레코드 리스트}.
    정답 리뷰어가 0명인 쿼리는 평균에서 제외하고, 쿼리마다 같은 비중으로 평균낸다."""
    summary, rows_all = {}, []
    for kind, records in answers.items():
        rows = []
        for a in records:
            qid = int(a["query_id"])
            cand = sorted(a["score"], key=int)
            labels = [a["score"][r] for r in cand]
            res = compute_ranking_metrics(labels, scores.loc[qid, cand].to_numpy())
            if sum(labels) > 0:
                rows.append(res)
            rows_all.append({"dataset": kind, "query_id": qid, "positive_count": int(sum(labels)), **res})
        summary[kind] = {**pd.DataFrame(rows).mean().to_dict(), "queries": len(rows)}
    return summary, pd.DataFrame(rows_all)
