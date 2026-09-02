# isThereEmotion

> **TRACE에 감정이라고 부를 만한 구조가 있을까요?**

EmoNet이 감정 대화 입력을 처리할 때 남긴 **RAW 뉴런 TRACE 자체를 데이터 분석하여**, 감정 정보를 보지 않은 상태에서도 안정적인 내부 상태 군집이 형성되는지 확인하는 프로젝트입니다.

이 저장소는 학교 **데이터 인사이트 기반 문제해결 수행평가**에 맞추어 구성했습니다. 분석 코드는 여러 Python 파일로 나누지 않고 **`isThereEmotion.ipynb` 하나에 모두 포함**되어 있습니다.

---

## 1. 문제 정의

### 연구 질문

> **EmoNet의 RAW 뉴런 TRACE를 감정 정보 없이 분석했을 때 안정적인 내부 상태 군집이 발견되는가? 또한 발견된 군집은 외부 감정 범주와 우연 이상의 연관성을 가지는가?**

이 프로젝트의 목적은 감정 label을 잘 맞히는 분류기를 만드는 것이 아닙니다.

먼저 감정 label을 완전히 사용하지 않은 상태에서 TRACE 자체의 구조를 찾고, 그 구조가 반복 분석에서도 유지되는지 확인합니다. 그 뒤에만 원 데이터의 감정 label을 공개하여 발견된 군집과 감정 사이의 관계를 검증합니다.

따라서 증거의 순서는 다음과 같습니다.

```text
RAW TRACE
    ↓
감정 label을 사용하지 않은 특징 추출
    ↓
비지도 군집 발견
    ↓
군집 품질 및 안정성 확인
    ↓
────────────────────────────
여기까지 감정 label 봉인
────────────────────────────
    ↓
감정 label 공개
    ↓
군집 ↔ 감정 연관성 검증
    ↓
Permutation test로 우연 여부 확인
```

핵심은 **감정이라는 정답을 먼저 보고 그에 맞는 군집을 만드는 것이 아니라**, TRACE에서 독립적으로 발견된 구조가 사후적으로 감정과 연결되는지를 확인하는 것입니다.

---

# 2. 데이터 수집

## 데이터 출처

기존 EmoNet 실험에서 생성한 `trajectory_batch_matrix120_v1`의 RAW TRACE를 이 저장소로 독립 이전했습니다.

- sample 수: **120개**
- 형식: JSON
- 위치: `data/raw/<sample_id>/raw_trace.json`
- RAW 전체 크기: 약 **1.81 GB**
- 대용량 원본은 Git LFS로 관리
- `data/source_manifest.csv`에 각 파일의 sample ID, label, 원본 크기, SHA-256, 경로 기록

원본 이전 후 분석에 필요한 RAW 데이터가 이 저장소 내부에 있기 때문에, 분석 과정은 기존 EmoNet 저장소에 의존하지 않습니다.

## RAW TRACE의 주요 항목

```text
raw_trace.json
├─ input_meta
│  ├─ sample_id
│  ├─ label
│  ├─ talk_id
│  └─ ...
│
└─ ticks
   ├─ tick
   ├─ active_nodes
   ├─ edges_fired
   └─ node_states
      ├─ node_id
      ├─ neuron_type
      ├─ K
      └─ stim_vec
```

분석 입력에는 주로 다음 정보를 사용합니다.

| RAW 항목 | 의미 |
|---|---|
| `active_nodes` | 각 tick에서 활성화된 뉴런 ID |
| `edges_fired` | 해당 tick에서 발화된 연결 |
| `node_id` | 뉴런 식별자 |
| `neuron_type` | excitatory / inhibitory / modulatory |
| `K` | 뉴런 내부 상태값 |
| `stim_vec` | 4차원 자극 벡터 |
| tick 순서 | 내부 상태의 시간적 변화 |

---

# 3. 데이터 누수 방지

이 프로젝트에서 가장 중요한 원칙입니다.

군집을 생성할 때 **감정 label을 feature로 사용하지 않습니다.**

또한 다음과 같이 이미 EmoNet의 후단 처리 과정에서 감정 해석과 관련되어 생성된 값도 입력 feature에서 제외합니다.

```text
label                  ← 군집 생성 단계에서 봉인
latent z_*
neurotransmitter-like outputs
dopamine / serotonin / ...
style s_*
predicted style s_hat_*
top_emotions
dominant_global_signal
LLM response
```

군집화에 사용하는 정보는 가능한 한 `ticks` 내부의 RAW 뉴런 상태에서 직접 계산합니다.

노트북 내부에서도 metadata와 feature를 별도 DataFrame으로 분리합니다.

```text
metadata_labels.csv
    └─ sample_id, label, talk_id, ...

trace_features.csv
    └─ sample_id + RAW TRACE에서 추출한 수치 특징
```

감정 label은 군집 수와 군집 구조를 확정한 이후의 **사후 검증 단계에서만** 사용합니다.

---

# 4. 데이터 전처리 계획

RAW JSON은 바로 군집 알고리즘에 넣을 수 없으므로 한 sample을 하나의 수치 feature vector로 변환합니다.

## 4-1. 계층형 JSON 통합

```text
raw_trace.json 여러 개
        ↓
tick / node 정보 추출
        ↓
sample 단위 feature 생성
        ↓
120 × N feature table
```

## 4-2. 품질 확인

감정과 무관한 기준으로 비정상 TRACE를 찾습니다.

현재 기준:

- tick 수가 3보다 작은 sample 제외
- 활성 뉴런이 한 번도 없는 sample 제외

이 기준은 label을 보지 않고 적용합니다.

## 4-3. 새로운 특징 생성

### 뉴런 활성 특징

- 평균 활성 뉴런 수
- 활성 비율 평균 / 표준편차
- 최대 활성 수
- 전체 활성량 AUC
- 초기 10 tick 활성 증가 기울기
- 초기/후기 활성 평균
- 최대 활성의 50%, 90%에 도달하는 tick
- unique active node 수와 비율
- 각 뉴런의 sample 내 활성 빈도 `node_freq_000 ~ node_freq_255`
- 뉴런 활성 지속성
- 뉴런별 binary entropy

### 연결 특징

- 평균/표준편차/최대 fired edge 수
- edge AUC
- 초기 edge 증가 기울기
- unique edge 수
- edge 재사용 비율
- active node 대비 edge 발화량

### 시간적 구조

- 연속 tick의 active-node set Jaccard similarity 평균/표준편차
- 전체 및 초기 변화 기울기

### 뉴런 종류

- excitatory state 비율
- inhibitory state 비율
- modulatory state 비율

### 내부 상태

- `K`의 평균/표준편차/시간 기울기
- neuron type별 `K` 통계
- `stim_vec` 각 축의 평균/표준편차/시간 기울기

## 4-4. K 스케일 처리

RAW에서 `K` 값의 범위가 매우 커질 수 있으므로 그대로 거리 계산에 사용하지 않습니다.

노트북에서는 부호를 보존하는 log 변환을 사용합니다.

```text
K' = sign(K) × log(1 + |K|)
```

## 4-5. 결측치 및 저분산 변수

- 수치형 결측치는 feature 중앙값으로 대치
- 거의 모든 sample에서 동일한 값인 저분산 변수 제거

## 4-6. 정규화

서로 다른 단위의 feature가 거리 계산을 지배하지 않도록 Z-score 표준화를 적용합니다.

```text
z = (x - mean) / std
```

## 4-7. PCA

고차원 TRACE feature를 그대로 거리 계산에 사용하면 noise와 차원의 영향을 크게 받을 수 있으므로 NumPy SVD를 이용해 PCA를 직접 구현합니다.

누적 설명분산 **90%**를 만족하는 principal components만 사용합니다.

---

# 5. 데이터 탐색 계획

감정 label을 보지 않고 다음을 확인합니다.

- sample 수와 RAW 품질
- 각 feature의 평균, 표준편차, 범위
- 저분산 feature 존재 여부
- 주요 변수의 분포
- PCA의 explained variance ratio

사용 가능한 분석 라이브러리가 NumPy와 pandas로 제한되어 있으므로 외부 plotting package를 사용하지 않습니다.

주요 feature 분포는 NumPy histogram을 이용한 **ASCII histogram**과 표로 확인합니다.

---

# 6. 알고리즘 1 — K-means Clustering

### 선정 이유

TRACE에서 추출한 수치형 내부 상태들이 거리 공간에서 자연스럽게 몇 개의 중심 주변으로 모이는지 확인하기 위해 사용합니다.

외부 머신러닝 라이브러리 없이 NumPy로 직접 구현합니다.

- K-means++ initialization
- 여러 초기화 중 inertia가 가장 작은 결과 선택
- `k = 2 ~ 10` 비교

---

# 7. 알고리즘 2 — Average-Linkage Hierarchical Clustering

### 선정 이유

K-means는 중심 기반 군집이라는 특정 가정을 사용합니다.

따라서 sample 간 거리에서 출발하여 가까운 군집을 순차적으로 합치는 **다른 원리의 계층적 군집화**를 함께 사용합니다.

두 알고리즘이 비슷한 구조를 발견한다면 특정 알고리즘 때문에 우연히 만들어진 군집이라는 설명이 약해집니다.

Average linkage 역시 NumPy만으로 직접 구현합니다.

---

# 8. 군집 수 및 군집 품질 평가

## Silhouette Score

각 `k=2~10`에 대해 Silhouette score를 계산합니다.

```text
a(i) = 같은 군집 sample과의 평균 거리
b(i) = 가장 가까운 다른 군집과의 평균 거리

s(i) = (b(i) - a(i)) / max(a(i), b(i))
```

K-means와 Hierarchical clustering 각각의 Silhouette를 구한 뒤, **두 알고리즘 평균 Silhouette가 가장 높은 k를 K\***로 선택합니다.

이 단계에서도 emotion label은 사용하지 않습니다.

---

# 9. 군집 안정성 검증

군집이 한 번만 나타났다는 사실은 강한 증거가 아닙니다.

따라서 다음을 추가로 확인합니다.

## 9-1. 초기화 변화

K-means의 random seed를 여러 번 바꾸어도 유사한 군집이 만들어지는지 확인합니다.

## 9-2. 작은 feature perturbation

PCA feature에 작은 5% noise를 추가해도 군집 구조가 유지되는지 확인합니다.

## ARI

군집 번호가 서로 달라도 구조 자체를 비교할 수 있도록 **Adjusted Rand Index**를 직접 구현해 사용합니다.

또한 같은 `K*`에서 K-means 결과와 Hierarchical 결과의 ARI도 계산합니다.

---

# 10. Negative Control

각 PCA 축의 값 분포는 유지하면서 sample 간 feature 조합을 독립적으로 shuffle합니다.

```text
실제 TRACE feature
      ↓
feature-wise shuffle
      ↓
sample 내부의 공동 구조 파괴
```

shuffle된 데이터에서 나타나는 Silhouette 분포와 실제 Silhouette를 비교합니다.

실제 군집 품질이 control보다 지속적으로 높다면 단순한 feature 범위나 주변분포 때문에 군집이 생긴 것이라는 설명을 약화시킬 수 있습니다.

---

# 11. 🔓 감정 label 사후 검증

**여기까지 군집 구조와 K\*를 먼저 고정한 뒤 감정 label을 공개합니다.**

## 검증 1 — Cluster ↔ Emotion NMI

발견된 cluster와 원 감정 E-code 사이의 Normalized Mutual Information을 계산합니다.

단, 현재 데이터는 감정 class 수가 많고 일부 class의 sample 수가 적으므로 NMI 절대값만으로 결론내리지 않습니다.

## 검증 2 — Label Permutation Test

감정 label의 개수 분포는 그대로 유지하면서 label 위치만 무작위로 섞습니다.

이를 여러 번 반복해 우연히 얻을 수 있는 NMI 분포를 만듭니다.

```text
실제 NMI = T_obs

shuffle 1 → T_1
shuffle 2 → T_2
...
shuffle B → T_B
```

경험적 p-value:

```text
p = (1 + count(T_perm >= T_obs)) / (B + 1)
```

실제 NMI가 permutation null보다 충분히 높다면 군집과 감정의 대응이 단순 우연이라고 보기 어려워집니다.

---

# 12. 독립적인 보강 증거

특정 clustering 알고리즘의 결과만으로 감정 구조를 주장하지 않기 위해 전체 TRACE 공간에서도 직접 확인합니다.

모든 sample 쌍에 대해 거리를 구해

```text
D_same       = 같은 감정 label sample 사이 거리
D_different  = 다른 감정 label sample 사이 거리
```

를 비교합니다.

관심 통계량은

```text
D_different - D_same
```

입니다.

양수라면 같은 감정끼리 상대적으로 더 가깝다는 뜻입니다.

이 값 역시 label permutation test로 우연 여부를 검증합니다.

---

# 13. 결과 해석 기준

이 프로젝트에서는 단순히 “군집이 보인다”는 이유로 감정이라고 부르지 않습니다.

| 관찰 결과 | 해석 |
|---|---|
| 뚜렷한 군집이 없음 | 현재 TRACE에 안정적인 잠재 상태 구조가 있다는 증거 부족 |
| 군집은 있으나 불안정 | 우연한 초기화/데이터 구조일 가능성 |
| 안정적인 군집이 있지만 감정과 관계 없음 | 내부 상태 구조는 있으나 감정 구조라고 부를 증거 부족 |
| 안정적인 군집 + 알고리즘 간 재현 + 감정과 permutation보다 강한 연관 | 감정과 체계적으로 연관된 잠재 TRACE 구조의 증거 |
| 위 결과 + 같은 감정끼리 거리도 유의하게 가까움 | clustering 방법에 독립적인 추가 보강 증거 |

### 주장 범위

결과가 강하게 나오더라도

> “EmoNet이 실제로 감정을 느낀다”

라고 주장하지 않습니다.

가능한 결론은 다음 수준입니다.

> **감정 정보를 사용하지 않은 비지도 분석에서 EmoNet의 내부 TRACE가 안정적인 잠재 상태 구조를 형성하고, 해당 구조가 외부 감정 범주와 무작위보다 강하게 연관된다면 TRACE 내부에 감정과 체계적으로 관련된 정보 구조가 존재한다는 증거로 볼 수 있다.**

---

# 14. 수행평가 항목 대응

| 수행평가 요구 | 프로젝트 계획 |
|---|---|
| 문제 정의 및 목적 | RAW TRACE에서 감정 관련 잠재 군집이 발견되는지 검증 |
| 데이터 수집 경로/방법 | EmoNet 실험 RAW TRACE 120개를 독립 저장소로 이전 |
| 데이터 내용/특성 | tick, active nodes, fired edges, neuron state, K, stim vector |
| 결측/이상 데이터 처리 | 실패 TRACE QC, 결측 중앙값 처리 |
| 데이터 통합 | 중첩 JSON 120개 → sample feature table |
| 데이터 삭제 | 후단 감정 출력 및 leakage 변수 제외 |
| 새로운 열 생성 | 활성/edge/시간/E-I-M/K/stim/node-frequency 특징 생성 |
| 정규화 | log 변환 + Z-score |
| 탐색 | 분포, PCA explained variance |
| 알고리즘 1 | K-means |
| 알고리즘 2 | Average-linkage Hierarchical Clustering |
| 알고리즘 선정 이유 | 서로 다른 원리에서 군집 구조 재현 확인 |
| 평가 지표 | Silhouette, ARI |
| 추가 신뢰도 검증 | negative control, NMI permutation, distance permutation |
| 활용 | TRACE에 감정 관련 잠재 구조가 존재하는지 판단 |

---

# 15. 실행

분석 코드 파일은 **하나뿐입니다.**

```text
isThereEmotion.ipynb
```

환경 설치:

```bash
git clone https://github.com/aether0xff-afk/isThereEmotion.git
cd isThereEmotion

git lfs pull
pip install -r requirements.txt
```

그 뒤 Jupyter 환경에서 `isThereEmotion.ipynb`를 위에서부터 순서대로 실행합니다.

## 사용 라이브러리

```text
numpy
pandas
```

분석 알고리즘은 `scikit-learn`을 사용하지 않고 NumPy/pandas로 직접 구현합니다.

---

# 저장소 구조

```text
isThereEmotion/
├─ isThereEmotion.ipynb       # 전체 분석 코드
├─ data/
│  ├─ raw/                    # 120개 원본 raw_trace.json (Git LFS)
│  ├─ source_manifest.csv     # 원본 파일 manifest
│  └─ SOURCE_COMMIT.txt       # 데이터 출처 commit 기록
├─ requirements.txt           # numpy, pandas
├─ README.md
└─ LICENSE
```

노트북을 실행하면 `data/processed/`와 `results/`가 생성되며, 전처리 feature와 군집/검증 결과 CSV가 저장됩니다.

---

## 핵심 한 문장

> **감정 label을 보지 않고 발견한 안정적인 RAW TRACE 군집이, label을 공개한 뒤에도 우연 이상의 감정 연관성을 보이는지를 검증한다.**
