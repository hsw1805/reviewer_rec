import argparse
from pathlib import Path

CODES = Path(__file__).resolve().parents[1]          # LightGCN_reviewer/codes
PROJECT = CODES.parent                               # LightGCN_reviewer


def none_or_int(v):
    return None if str(v).lower() in ("none", "null", "-1") else int(v)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="전처리한 인용 그래프 + LightGCN (+MLP, +연도) 리뷰어 추천")

    # 경로
    p.add_argument("--data_dir", default=str(PROJECT.parent / "dataset/kdd_2hop"),
                   help="edges.parquet, papers_normalized_v2.parquet, reviewer_papers.jsonl, test_answers_*.jsonl, embedding/")
    p.add_argument("--cache_dir", default=str(PROJECT / "cache"), help="그래프·valid·prompt 임베딩 캐시")
    p.add_argument("--save_dir", default=str(PROJECT / "runs"), help="결과 저장 폴더")
    p.add_argument("--model_name", default=None, help="결과 폴더 이름 (없으면 설정으로 자동)")

    p.add_argument("--model", choices=["lightgcn", "specter2"], default="lightgcn",
                   help="lightgcn: 전처리 그래프로 전파 / specter2: 전파 없이 초기 임베딩 그대로 (그래프·MLP 인자 무시)")

    # 1) input graph 전처리
    p.add_argument("--sim_percent", type=int, default=100, help="인용 edge 중 source별 SPECTER2 cos 상위 몇 %%를 남길지 (25/50/75/100)")
    p.add_argument("--year_gap", type=none_or_int, default=None, help="양 끝 논문 연도 차가 n년 이하인 edge만 (none = 제한 없음)")

    # 2) 모델
    p.add_argument("--mlp", type=int, choices=[0, 1], default=0,
                   help="0: 초기 임베딩을 그대로 전파 (학습 없음) / 1: MLP → L2 정규화 → 전파, MLP를 BPR로 학습")
    p.add_argument("--year_mode", type=int, choices=[0, 1, 2, 3], default=0,
                   help="0: 연도 없음 / 1: prompt year / 2: scalar year / 3: Time2Vec year")
    p.add_argument("--prompt_path", default=None, help="year_mode 1의 임베딩 파일 (기본: cache_dir/prompt_year_embeddings.pt)")
    p.add_argument("--n_layers", type=int, default=2, help="LightGCN 전파 층 수")
    p.add_argument("--hidden", type=int, default=256, help="MLP 은닉 차원")
    p.add_argument("--dim", type=int, default=64, help="MLP 출력 차원")
    p.add_argument("--t2v_dim", type=int, default=17, help="Time2Vec 차원")

    # 3) 리뷰어 점수
    p.add_argument("--topk", type=int, default=1, help="리뷰어 과거 논문 중 cos 상위 k편 평균 (1 = max, 0 = 1~5 전부)")
    p.add_argument("--review_year", type=none_or_int, default=None,
                   help="리뷰어 과거 논문 중 2020−n년 이후 논문만 사용 (3/4/5, none = 자르지 않음, 0 = none·3·4·5 전부)")

    # 4) 학습 (mlp 1일 때만)
    p.add_argument("--train_sim_percent", type=int, default=25, help="positive edge 그래프의 유사도 상위 %%")
    p.add_argument("--train_year_gap", type=none_or_int, default=5, help="positive edge 그래프의 최대 연도 차")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=16384)
    p.add_argument("--tau", type=float, default=0.05, help="BPR 점수 = cos / tau")
    p.add_argument("--max_epochs", type=int, default=200)
    p.add_argument("--eval_every", type=int, default=5, help="몇 에폭마다 valid를 잴지")
    p.add_argument("--patience", type=int, default=10, help="valid 개선이 없는 평가 횟수가 이만큼이면 멈춤")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu_id", type=int, default=0)
    args = p.parse_args(argv)
    if args.prompt_path is None:
        args.prompt_path = str(Path(args.cache_dir) / "prompt_year_embeddings.pt")
    if args.model == "specter2" and args.mlp:
        p.error("--model specter2 는 전파·학습이 없어서 --mlp 1 을 쓸 수 없다")
    args.topks = [1, 2, 3, 4, 5] if args.topk == 0 else [args.topk]
    args.review_years = [None, 3, 4, 5] if args.review_year == 0 else [args.review_year]
    if args.model_name is None:
        k = "all" if args.topk == 0 else args.topk
        n = "all" if args.review_year == 0 else args.review_year
        if args.model == "specter2":
            args.model_name = f"specter2_year{args.year_mode}_k{k}_n{n}"
        else:
            args.model_name = (f"lightgcn_mlp{args.mlp}_year{args.year_mode}_s{args.sim_percent}_y{args.year_gap}"
                               f"_k{k}_n{n}" + (f"_seed{args.seed}" if args.mlp else ""))
    return args
