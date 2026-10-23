# reviewer_rec

## 리뷰어 추천시스템 코드 정리

### LightGCN 리뷰어 추천 (전처리 그래프 + MLP + 연도)

쿼리 논문에 맞는 리뷰어를 후보 20명 중에서 순위 매긴다. SPECTER2 임베딩을 전처리한 인용 그래프 위에서 LightGCN으로 전파하고, 리뷰어 점수는 과거 논문과의 cos으로 낸다.

```
SPECTER2 (+연도) ──[MLP]──▶ E0 ──LightGCN 전파 (전처리한 인용 그래프)──▶ H ──▶ 리뷰어 점수 = (최근 n년) 과거 논문 cos 상위 k편 평균
```

`--model specter2`는 전파 없이 E0 그대로 리뷰어 점수를 낸다 (기준선).

### LightGCN + GCL (kNN view × 인용 그래프 view 대조학습)

`--model lightgcn+gcl`. MLP 없이 초기 임베딩 테이블을 직접 학습한다. SPECTER2 kNN 그래프와 전처리한 인용 그래프를 두 view로 GRACE 대조학습을 하고, 학습된 테이블을 input graph로 전파한다.

```
view1: T1 = E0 (고정)          ──kNN 그래프 (knn_k)──────────────▶ LightGCN ──▶ z1
view2: T2 = E0로 시작 (학습)    ──view2 그래프 (view2_sim/year)──▶ LightGCN ──▶ z2
L = GRACE( proj(z1), proj(z2) ),  proj = 2층 projection (두 view 공유)
최종 H = T2 ──LightGCN 전파 (input graph)──▶ H ──▶ 리뷰어 점수 (위와 같음)
```

## 구조

```
LightGCN_reviewer/
├── README.md
├── cache/                    # 그래프·valid·prompt 임베딩 캐시 (처음 실행할 때 자동 생성)
├── runs/<model_name>/        # 결과: metrics.csv (조합 × 세트 × 지표 10개), metrics.json, per_query.csv, curve.csv(학습 시)
└── codes/
    ├── main.py               # 실행: 그래프 → 모델 → (학습) → 평가
    ├── Models.py             # LightGCN, LightGCNGCL(+projection, GRACE loss), Time2Vec, 정규화 인접행렬
    ├── data/build_data.py    # input graph 전처리, kNN 그래프, valid 세트, prompt year 임베딩 생성
    └── utility/
        ├── parser.py         # 하이퍼파라미터
        ├── load_data.py      # 임베딩·연도·리뷰어·정답 읽기
        ├── metrics.py        # 지표 10개
        └── batch_test.py     # 리뷰어 점수(top-k), 평가, valid
```

다른 프로젝트 파일(`evaluation.py` 등) 없이 이 폴더와 데이터 폴더만 있으면 돌아간다. 환경: conda `reviewer_rec`.

## 실행

`codes/` 폴더에서 실행한다.

```bash
cd codes
python main.py --model specter2 --topk 0 --review_year 0                                  # SPECTER2, top-k 1~5 × 연도 none/3/4/5
python main.py --model lightgcn --sim_percent 100 --year_gap none --mlp 0 --year_mode 0   # 전파만
python main.py --model lightgcn --sim_percent 75 --year_gap 3 --mlp 0 --year_mode 2 --topk 3 --review_year 3
python main.py --model lightgcn --sim_percent 100 --year_gap none --mlp 1 --year_mode 3 --topk 0 --review_year 0   # MLP 학습
python main.py --model lightgcn+gcl --knn_k 100 --view2_sim_percent 50 --view2_year_gap 5 --year_mode 3 --topk 0 --review_year 0   # GCL 학습
```

kNN 그래프는 처음 실행할 때 `cache/knn_k{k}.npy`로 만들어진다 (`python data/build_data.py --knn --knn_k 100`으로 미리 만들 수도 있다).
그리드는 인자를 바꿔 여러 번 실행한다. 예: `for k in 50 100 150 200; do for s in 100 75 50; do python main.py --model lightgcn+gcl --knn_k $k --view2_sim_percent $s --view2_year_gap 5 --year_mode 3 --topk 0 --review_year 0; done; done`

## 하이퍼파라미터

| 인자 | 기본값 | 의미 |
|---|---|---|
| **`--model`** | lightgcn | lightgcn: 전처리 그래프로 전파 / lightgcn+gcl: GCL로 E0 테이블을 학습한 뒤 전파 (`--mlp 1` 불가) / specter2: 전파 없이 E0 그대로 (그래프·MLP 인자 무시, `--mlp 1` 불가) |
| **`--sim_percent`** | 100 | input graph 전처리: 인용 edge 중 인용한 논문별 SPECTER2 cos 상위 몇 %를 남길지 |
| **`--year_gap`** | none | input graph 전처리: 양 끝 논문 연도 차가 n년 이하인 edge만 (`none` = 제한 없음) |
| **`--mlp`** | 0 | 0: 초기 임베딩을 그대로 전파 (학습 없음) / 1: MLP를 거쳐 전파, MLP 학습 |
| **`--year_mode`** | 0 | 0: 연도 없음 / 1: prompt year / 2: scalar year / 3: Time2Vec year |
| **`--topk`** | 1 | 리뷰어 점수 = 과거 논문 중 cos 상위 k편 평균 (1 = max). **0이면 1~5 전부** |
| **`--review_year`** | none | 리뷰어 과거 논문 중 2020−n년 이후 논문만 사용 (3/4/5). none = 자르지 않음. **0이면 none·3·4·5 전부** |
| `--n_layers` | 2 | LightGCN 전파 층 수 |
| `--hidden`, `--dim` | 256, 64 | MLP 은닉·출력 차원 (`--mlp 1`) |
| `--t2v_dim` | 17 | Time2Vec 차원 (`--year_mode 3`) |
| `--prompt_path` | cache/prompt_year_embeddings.pt | `--year_mode 1`의 임베딩 파일 |
| `--train_sim_percent`, `--train_year_gap` | 25, 5 | 학습 positive edge 그래프 (`--mlp 1`) |
| `--lr`, `--batch`, `--tau` | 1e-3, 16384, 0.05 | Adam 학습률, 배치(쌍), BPR 온도 |
| `--max_epochs`, `--eval_every`, `--patience` | 200, 5, 10 | 최대 에폭, valid 측정 주기, 얼리스타핑 |
| **`--knn_k`** | 100 | view1 kNN 그래프: 논문마다 SPECTER2 cos 상위 k편 (`lightgcn+gcl`) |
| **`--view2_sim_percent`**, **`--view2_year_gap`** | 50, 5 | view2 인용 그래프 (input graph 전처리와 같은 방식, `lightgcn+gcl`) |
| `--gcl_steps`, `--gcl_batch`, `--gcl_lr`, `--gcl_tau` | 100, 8192, 1e-3, 0.5 | GCL 학습 step 수, step마다 뽑는 노드 수, Adam 학습률, GRACE 온도 |
| `--seed`, `--gpu_id` | 0, 0 | |
| `--data_dir`, `--cache_dir`, `--save_dir`, `--model_name` | | 경로. 결과 폴더 이름은 기본으로 설정에서 자동 생성 (`mlp1_year3_s100_yNone_k1_seed0`, `lightgcn_gcl_year3_s100_yNone_knn100_v2s50_y5_kall_nall_seed0`) |

## 각 단계

### 1. input graph 전처리 (`data/build_data.py: build_graph`)
1. 인용 edge에서 중복·자기 인용을 지우고, 양 끝 논문 모두 임베딩과 연도가 있는 edge만 남긴다.
2. `sim_percent < 100`이면 인용한 논문마다 참고문헌을 SPECTER2 cos 순으로 정렬해 상위 ceil(n × sim_percent %)개만 남긴다.
3. `year_gap`이 있으면 |연도 차| ≤ year_gap인 edge만 남긴다.
4. 무방향으로 바꾸고 Â = D^-1/2 A D^-1/2로 정규화한다. 결과는 `cache/graph_s{sim}_y{gap}.npy`에 저장한다.

### 2. 초기 임베딩 E0와 연도 (`Models.py: LightGCN`)

입력 특징 X = [연도 특징, 텍스트 임베딩]

| year_mode | 연도 특징 | 텍스트 임베딩 |
|---|---|---|
| 0 | 없음 | SPECTER2 (768) |
| 1 prompt | 없음 | `title [SEP] Year: {year}. abstract`로 다시 인코딩한 SPECTER2 (768) |
| 2 scalar | τ = (clip(year, 1970, 2026) − 2008) / 5 (1) | SPECTER2 |
| 3 Time2Vec | [ω0·τ + φ0, sin(ω_i·τ + φ_i)] (17) | SPECTER2 |

- 연도 없는 논문은 τ = 0(Time2Vec은 0벡터). 연도 없는 쿼리 7편(58, 90, 96, 106, 132, 134, 163)은 2020년으로 둔다.
- `--mlp 0`: E0 = X. 학습이 없어서 Time2Vec의 ω, φ도 seed로 고정한 초기값 그대로다.
- `--mlp 1`: E0 = normalize(Linear(256→64)(ReLU(Linear(X→256)))). 텍스트 임베딩은 고정하고 MLP(와 Time2Vec ω, φ)만 학습한다.
- prompt 임베딩은 `python data/build_data.py --prompt`로 만든다(89만 편을 다시 인코딩, GPU로 수 시간). 지금 `cache/`의 파일은 아이디어 03에서 만든 파일에 걸어 둔 링크다.

### 3. LightGCN 전파
H = (E0 + ÂE0 + Â²E0) / 3 (`n_layers = 2`). 별도의 변환 행렬·비선형 없이 이웃 평균만 한다.

### 4. 학습 (`--mlp 1`, `main.py: train`)
- positive: train graph(`train_sim_percent`, `train_year_gap`, 기본 s25_y5)의 edge. valid 가짜 쿼리에 닿는 edge와 쿼리를 인용한 edge(쿼리보다 나중 논문)는 뺀다.
- negative: anchor마다 train 노드 중 무작위 1개(인용 edge가 있는 쌍은 다시 뽑는다).
- loss: BPR, 점수 = cos(H_a, H_b) / τ. Adam, 1 에폭 = positive 전체를 한 번.
- 체크포인트 선택: `eval_every` 에폭마다 valid MRR을 재고, `patience`번 연속 개선이 없으면 멈춘다. valid 최고 에폭의 모델로 평가한다.
  valid = 리뷰어 과거 논문 3,000편(2018~2020)을 가짜 쿼리로 써서, 그 논문을 숨긴 이력으로 후보 20명 중 저자를 맞히는 과제. **평가 정답은 선택에 쓰지 않는다.**

### 5. 리뷰어 점수와 평가 (`utility/batch_test.py`, `utility/metrics.py`)
- ① `review_year` n이 있으면 리뷰어 과거 논문 중 연도 ≥ 2020 − n인 논문만 남긴다 (연도 없는 논문은 뺀다). 남은 논문이 없는 리뷰어는 최하위 점수(−10).
- ② 남은 과거 논문 중 쿼리와 cos 상위 `topk`편의 평균 (k편보다 적으면 남은 만큼 평균).
- `--topk 0`, `--review_year 0`이면 임베딩은 한 번만 만들고 조합마다 점수·평가를 따로 낸다.
- 데이터 폴더의 `test_answers_*.jsonl`을 전부 평가한다(soft, hard, soft/hard_25y3·25y4·25y5). 정답 리뷰어 0명인 쿼리만 평균에서 뺀다.
- 지표 10개: P@1~5, P@10, NDCG@5, NDCG@10, MRR, AUC. 동점은 리뷰어 ID 순, P@k 분모는 항상 k.

### 6. GCL 학습 (`--model lightgcn+gcl`, `main.py: train_gcl`, `Models.py: LightGCNGCL`)
- view1 그래프 (`data/build_data.py: build_knn`): 논문마다 SPECTER2 cos 상위 `knn_k`편(자기 자신 제외)을 무방향으로 합친다. 양 끝 중 연도 없는 논문이 있는 쌍은 뺀다(인용 그래프 ①과 같은 기준, 쿼리 7편은 2020년으로 보므로 남는다).
- view2 그래프: `build_graph(view2_sim_percent, view2_year_gap)`. input graph 전처리와 같은 방식이다.
- T1 = E0 고정, T2 = E0으로 시작해서 학습. year_mode 2는 τ 열까지 T2에 넣어 학습하고, year_mode 3은 Time2Vec ω, φ를 두 view가 공유해 학습한다.
  z1 = LightGCN(kNN, T1)은 한 번만 계산한다. Time2Vec 블록은 연도에만 달려 있어서, 연도 one-hot을 kNN으로 미리 전파한 M으로 M · T2V(연도값)을 구한다(매 step 전파하는 것과 같다).
- step마다 연도 있는 논문 중 무작위 `gcl_batch`개를 뽑아 z1, z2 = LightGCN(view2, T2)를 만들고, 공유 projection(Linear → ELU → Linear)을 거친 GRACE NT-Xent loss(positive = 같은 노드의 두 view, negative = batch 안 다른 노드, τ = `gcl_tau`)로 T2(+Time2Vec)와 projection을 학습한다. Adam, `gcl_steps` step.
- valid·얼리스타핑 없이 마지막 step의 T2를 input graph(`sim_percent`, `year_gap`, 기본 s100_yNone = 전처리한 원본 인용 그래프)로 전파한 H로 평가한다. projection은 평가에 쓰지 않는다.
- `curve.csv`에 step별 loss를 남기고, `metrics.json`에 첫·마지막 loss와 학습 후 T2와 E0의 cos 중앙값(얼마나 바뀌었는지)을 남긴다.

## 결과 (kdd_2hop, 학습 없음)

`runs/specter2_year0_kall_nall`, `runs/lightgcn_mlp0_year0_s100_yNone_kall_nall`. P@5, 연도 자르지 않음(review_year none):

| 모델 | topk | soft | hard | soft_25y3 | hard_25y3 | soft_25y4 | hard_25y4 | soft_25y5 | hard_25y5 |
|---|---|---|---|---|---|---|---|---|---|
| SPECTER2 | 1 | 0.3552 | 0.1770 | 0.3743 | 0.2438 | 0.3806 | 0.2424 | 0.3865 | 0.2478 |
| SPECTER2 | 4 | 0.3828 | 0.1905 | 0.3842 | 0.2562 | 0.3942 | 0.2606 | 0.4038 | 0.2687 |
| LightGCN s100 | 1 | 0.3621 | 0.1865 | 0.3525 | 0.2344 | 0.3592 | 0.2333 | 0.3692 | 0.2418 |
| LightGCN s100 | 4 | 0.3793 | 0.1919 | 0.3683 | 0.2344 | 0.3748 | 0.2333 | 0.3788 | 0.2388 |

top-k 1~5 × 연도 none/3/4/5 전체, P@10과 지표 10개는 각 폴더의 `metrics.csv`. 위 수치는 평가 정답으로 설정을 고른 것이 아니라 조합별 결과를 그대로 보인 것이다.

## 확인한 것 (kdd_2hop)

| 설정 | soft P@5 | hard P@5 | 비교 |
|---|---|---|---|
| `--model lightgcn --mlp 0 --year_mode 0` (s100) | 0.3621 | 0.1865 | `홍세원.py`와 같음 |
| `--model specter2 --topk 1 --review_year 3` | 0.3494 | 0.1743 | `홍세원.py --recent_years 3`과 같음 (soft_25y3 0.4079) |
| `--mlp 0 --year_mode 1/2/3` | 0.3632 / 0.3644 / 0.3667 | 0.1892 / 0.1905 / 0.1919 | `홍세원.py`와 같음 |
| `--mlp 1 --year_mode 0` | 학습 전 valid 0.8625, 1 에폭 loss 0.0055 | | 아이디어 08 1번과 같음 |

그래프 s75_yNone, s25_y5와 valid 세트는 아이디어 02·03의 캐시와 똑같이 만들어진다.

## 결과: LightGCN + GCL (kdd_2hop)

`runs/lightgcn_gcl_year3_s100_yNone_knn100_v2s50_y5_kall_nall_seed0` (기본값: kNN k100, view2 s50_y5, year_mode 3, input graph s100_yNone, 100 step, seed 0). P@5, 연도 자르지 않음(review_year none):

| 모델 | topk | soft | hard | soft_25y3 | hard_25y3 | soft_25y4 | hard_25y4 | soft_25y5 | hard_25y5 |
|---|---|---|---|---|---|---|---|---|---|
| LightGCN year3 (학습 없음) | 1 | 0.3667 | 0.1919 | 0.3604 | 0.2406 | 0.3670 | 0.2394 | 0.3750 | 0.2478 |
| LightGCN year3 (학습 없음) | 4 | 0.3816 | 0.1905 | 0.3683 | 0.2312 | 0.3748 | 0.2303 | 0.3788 | 0.2358 |
| LightGCN+GCL year3 | 1 | 0.3667 | 0.1919 | 0.3584 | 0.2375 | 0.3650 | 0.2364 | 0.3731 | 0.2448 |
| LightGCN+GCL year3 | 4 | 0.3839 | 0.1946 | 0.3703 | 0.2344 | 0.3767 | 0.2333 | 0.3808 | 0.2388 |

- 학습 없음 행은 같은 input graph의 `--model lightgcn --mlp 0 --year_mode 3`이다(같은 환경에서 다시 돌림, 위 확인 표의 값과 같다).
- loss는 9.44 → 7.82 (100 step)로 내려가지만, 학습 후 T2와 E0의 cos 중앙값이 1.0000이다. 테이블이 거의 바뀌지 않아서 학습 없음과 차이가 칸마다 ±0.005 이내(쿼리 1~2개)다.
- 실행 시간: GPU 1장(RTX 5000 Ada 32GB) 기준 kNN 그래프를 만든 뒤 약 1분 30초. 학습 파라미터는 6.85억 개(T2 테이블 89만 × 768 + Time2Vec + projection)다.
- 이 코드는 `knn_gcl_gird_search`(별도 데이터 폴더의 kNN × 엣지 필터링 GCL 그리드 서치)를 이 저장소의 데이터·그래프 전처리·연도 방식·평가에 맞춰 옮긴 것이다. 그쪽과 달라진 점: view2 유사도 필터가 전체 edge 상위 %가 아니라 인용한 논문별 상위 %다(`build_graph`). 연도 없는 논문은 지우지 않고 τ = 0으로 두며(edge는 똑같이 빠진다), 연도 방식·Time2Vec 초기값은 이 저장소의 `year_mode`를 따른다. 그래서 수치가 그쪽 결과표와 같지는 않다.
