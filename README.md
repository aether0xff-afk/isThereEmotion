# isThereEmotion

> **TRACE에 정말 감정이라고 부를 만한 구조가 있을까?**

EmoNet이 감정 대화 입력을 처리하면서 남긴 **RAW 뉴런 TRACE**를 데이터 과학적으로 분석하는 독립 프로젝트입니다.

이 프로젝트의 목적은 감정 라벨을 잘 맞히는 분류기를 만드는 것이 아닙니다. 먼저 감정 정보를 보지 않은 상태에서 TRACE 내부에 **자연스럽고 안정적인 군집**이 존재하는지 확인하고, 그 뒤에야 원 데이터의 감정 범주와 비교하여 그 군집이 감정과 **우연 이상의 관계**를 가지는지 검증합니다.

---

# 수행평가 1단계 — 문제 정의서

## 1. 문제 정의 및 목적

### 연구 질문

> **EmoNet의 RAW 뉴런 TRACE를 감정 정보 없이 분석했을 때 안정적인 내부 상태 군집이 발견되는가? 또한 발견된 군집은 외부 감정 범주와 우연 이상의 연관성을 가지는가?**

### 문제를 이렇게 정의한 이유

EmoNet은 하나의 감정 대화를 처리하는 동안 여러 tick에 걸쳐 뉴런 활성, 연결 발화, 내부 상태 변화를 남깁니다. 최종 감정 출력만 보면 내부에서 실제로 어떤 구조가 만들어졌는지는 알기 어렵습니다.

따라서 본 프로젝트에서는 다음 순서로 증거를 확인합니다.

```text
RAW TRACE
   ↓
감정 label을 사용하지 않은 전처리·특징 생성
   ↓
비지도 군집 발견
   ↓
군집 품질 및 안정성 확인
   ↓
마지막에만 실제 감정 label 공개
   ↓
군집과 감정의 연관성 검증
```

본 프로젝트에서 **감정과 관련된 잠재 군집의 증거**는 다음 세 조건으로 정의합니다.

1. 감정 label을 사용하지 않아도 군집이 형성된다.
2. 초기 조건이나 군집 방법을 바꾸어도 유사한 구조가 반복된다.
3. 만들어진 군집이 실제 감정 범주와 무작위보다 강하게 대응한다.

결과가 좋지 않더라도 분석은 의미가 있습니다. 군집이 형성되지 않으면 현재 TRACE에 뚜렷한 잠재 상태 구조가 없다는 근거가 되고, 군집은 존재하지만 감정과 관계가 없다면 그 구조를 감정 구조라고 해석하기 어렵다는 결론을 낼 수 있습니다.

---

## 2. 데이터 수집 경로·방법·내용

### 데이터 출처

기존 EmoNet 실험에서 생성된 `v4/outputs/research/trajectory_batch_matrix120_v1`의 RAW TRACE를 이 저장소로 이전했습니다.

분석에 필요한 원본은 이 저장소 안에 독립적으로 포함되어 있으므로 이후 분석은 EmoNet 저장소에 의존하지 않습니다.

```text
data/raw/<sample_id>/raw_trace.json
```

- RAW TRACE: **120 samples**
- 원본 전체 크기: 약 **1.81 GB**
- 대용량 JSON은 Git LFS로 관리
- `data/source_manifest.csv`에 sample별 label, 원본 크기, SHA-256, 경로 기록

### RAW TRACE의 주요 데이터

각 sample에는 여러 tick의 내부 상태가 저장되어 있습니다.

| 항목 | 의미 |
|---|---|
| `tick` | 내부 상태의 시간 단계 |
| `active_nodes` | 해당 tick에 활성화된 뉴런 목록 |
| `edges_fired` | 해당 tick에 발화한 연결 |
| `node_id` | 뉴런 식별자 |
| `neuron_type` | excitatory / inhibitory / modulatory |
| `K` | 뉴런 내부 상태값 |
| `stim_vec` | 뉴런에 입력된 자극 벡터 |
| `input_meta.label` | 원 데이터의 감정 범주 — **군집 생성에는 사용하지 않음** |

---

## 3. 데이터 전처리 계획

RAW TRACE는 바로 군집화할 수 있는 표 형태가 아니라 **sample → tick → node**로 중첩된 JSON입니다. 따라서 전처리 자체가 이 프로젝트의 중요한 분석 과정입니다.

### 3-1. 중첩 JSON 통합

```text
raw_trace.json 여러 개
        ↓
tick 단위 / node 단위 정보 추출
        ↓
sample 단위 특징으로 집계
        ↓
1 sample = 1 row의 분석용 DataFrame
```

현재 `scripts/01_build_features.py`가 120개의 RAW TRACE를 읽어 `data/processed/features.csv`를 생성합니다.

### 3-2. 감정 정보 누수 방지

군집이 감정 label에 의해 미리 유도되지 않도록 **군집 생성 단계에서는 label을 완전히 제외**합니다.

또한 다음과 같이 이미 EmoNet의 후단 처리 과정에서 만들어진 감정 해석·출력값은 군집 feature로 사용하지 않습니다.

- latent `z_*`
- dopamine / serotonin / norepinephrine / melatonin 등 후단 값
- style / predicted style 값
- `top_emotions`
- `dominant_global_signal`
- 생성된 LLM response

즉 군집의 입력은 가능한 한 **실제 뉴런 TRACE에서 직접 계산한 값만** 사용합니다.

### 3-3. 새로운 특징 생성

RAW 데이터에서 다음과 같은 수치형 특징을 만듭니다.

#### 뉴런 활성

- 평균 / 표준편차 / 최대 활성 뉴런 비율
- 초기 활성 증가 기울기
- early / late 활성 평균
- 최대 활성의 일정 비율에 도달하는 시점
- 각 뉴런의 전체 tick 대비 활성 빈도 `node_freq_***`
- 활성 뉴런의 지속성

#### 연결 구조

- 평균 / 표준편차 / 최대 edge 발화량
- unique edge 수
- edge 재사용률
- 연속 tick 사이 활성 뉴런 집합의 Jaccard similarity

#### 뉴런 종류

- excitatory 활성 비율
- inhibitory 활성 비율
- modulatory 활성 비율

#### 내부 상태 변화

- `K` 평균 / 표준편차 / 변화 기울기
- `stim_vec` 평균 / 표준편차 / 변화량

### 3-4. 이상치·스케일 처리

실제 RAW TRACE에서 `K`는 값의 범위가 매우 크기 때문에 그대로 거리 계산에 넣으면 다른 특징을 압도할 수 있습니다.

따라서 다음 로그 변환을 적용합니다.

```text
K' = log(1 + K)
```

또한 K-means와 같이 거리의 영향을 받는 알고리즘을 사용하므로 특징별 단위 차이를 줄이기 위해 Z-score 표준화를 적용합니다.

```text
z = (x - mean) / std
```

### 3-5. 정보량이 낮은 특징 제거

거의 모든 sample에서 같은 값을 가지는 저분산 특징은 군집 형성에 의미가 적으므로 분산을 확인한 뒤 제거합니다.

분석·전처리 코드는 수행평가 조건에 맞게 **pandas와 NumPy 중심으로 작성**합니다.

---

## 4. 데이터 탐색 및 시각화 계획

> 이 단계에서도 감정 label은 군집 형성이나 파라미터 선택에 사용하지 않습니다.

### 4-1. 특징 분포 확인

각 특징의 평균, 표준편차, 최소·최대값, 분위수를 확인하여 편향된 분포나 이상치를 찾습니다.

### 4-2. 특징 간 상관관계 확인

`pandas.DataFrame.corr()`를 이용해 서로 거의 같은 정보를 가지는 특징을 확인합니다. 지나치게 중복되는 열이 군집 거리를 과도하게 지배하지 않는지 점검합니다.

### 4-3. PCA를 이용한 저차원 구조 확인

NumPy의 SVD를 이용해 PCA를 직접 계산합니다.

고차원 TRACE 특징을 2차원 또는 소수의 주성분으로 축소하여 데이터 자체에 어떤 구조가 있는지 확인합니다.

중요한 점은 **처음 PCA 결과를 확인할 때 감정별 색을 사용하지 않는 것**입니다. 군집 분석과 파라미터 결정이 끝난 뒤에만 감정 label과 비교합니다.

---

## 5. 사용할 알고리즘 — 2개 이상

### 알고리즘 1. K-means Clustering

비슷한 TRACE 특징을 가진 sample을 거리 기반으로 묶습니다.

**선택 이유**

- 수치형 특징에 적용하기 단순하고 명확함
- 군집 중심과 sample 간 거리를 직접 해석할 수 있음
- NumPy만으로 직접 구현 가능
- `k = 2, 3, ...`을 비교하여 자연스러운 군집 수 후보를 탐색할 수 있음

단, **감정 label과 잘 맞는 k를 선택하지 않습니다.** 군집 내부 품질만으로 k를 결정합니다.

### 알고리즘 2. Hierarchical Clustering

sample 간 거리가 가까운 군집을 단계적으로 합쳐 전체 데이터의 계층 구조를 확인합니다.

**선택 이유**

- K-means와 다른 방식으로 군집 구조를 확인할 수 있음
- 특정 알고리즘에서만 우연히 나타난 군집인지 비교할 수 있음
- 두 알고리즘이 유사한 구조를 만들면 군집의 재현성에 대한 추가 근거가 됨

---

## 6. 알고리즘 평가 지표

이 프로젝트에서는 분류 Accuracy보다 **군집 자체의 품질과 안정성**이 중요합니다.

### 6-1. Silhouette Score

한 sample이 자기 군집에는 얼마나 가깝고 다른 군집과는 얼마나 떨어져 있는지를 측정합니다.

```text
s(i) = (b(i) - a(i)) / max(a(i), b(i))
```

- `a(i)` : 같은 군집 sample들과의 평균 거리
- `b(i)` : 가장 가까운 다른 군집 sample들과의 평균 거리
- 값이 클수록 군집이 잘 분리되어 있음

### 6-2. 군집 안정성

K-means의 초기 중심이나 random seed를 바꾸어 여러 번 실행합니다.

초기값이 달라도 비슷한 sample끼리 계속 묶이는지 확인하여 **한 번 우연히 나온 군집인지, 반복되는 내부 구조인지** 평가합니다.

필요하면 군집 결과 사이의 일치도를 ARI 등의 방식으로 계산합니다.

---

## 7. 감정과의 사후 검증

**군집 생성과 군집 수 선택이 모두 끝난 이후에 처음으로 감정 label을 사용합니다.**

### 7-1. Cluster × Emotion 교차표

`pandas.crosstab()`을 이용해 각 cluster에 실제 감정 범주가 어떻게 분포하는지 확인합니다.

이를 통해 특정 감정들이 특정 cluster에 집중되는지 분석합니다.

### 7-2. Permutation Test — 추가 검증

군집은 그대로 둔 상태에서 감정 label만 무작위로 여러 번 섞습니다.

```text
실제 cluster ↔ emotion 연관성
            vs
무작위 label ↔ cluster 연관성
```

실제 데이터의 연관성이 무작위 결과보다 지속적으로 크다면, 군집과 감정의 관계를 단순한 우연으로 설명하기 어려워집니다.

이 검정은 NumPy의 난수 셔플을 이용해 직접 수행할 계획입니다.

### 7-3. 독립적인 보강 분석

군집 알고리즘 자체와 무관하게 다음 거리도 비교합니다.

```text
D_same       = 같은 감정 sample 사이 평균 TRACE 거리
D_different  = 다른 감정 sample 사이 평균 TRACE 거리
```

만약 `D_same < D_different`이고 permutation test에서도 같은 경향이 확인된다면, 감정과 TRACE 구조의 관계에 대한 별도의 근거가 됩니다.

---

## 8. 분석 결과의 활용 및 기대 효과

본 프로젝트의 목적은 **EmoNet이 실제로 감정을 느낀다고 증명하는 것**이 아닙니다.

분석 결과에 따라 다음과 같이 판단합니다.

| 결과 | 해석 |
|---|---|
| 뚜렷한 군집이 발견되지 않음 | 현재 TRACE에 명확한 잠재 상태 군집이 있다는 증거가 부족함 |
| 안정적인 군집은 있으나 감정과 관계가 없음 | 내부 구조는 존재하지만 감정 구조라고 해석하기 어려움 |
| 안정적인 군집이 있고 감정과 우연 이상의 연관성을 보임 | TRACE 내부에 감정과 체계적으로 관련된 잠재 구조가 존재한다는 증거 |

따라서 최종 결론은 모델의 의도나 이름이 아니라 **RAW 데이터에서 실제로 관찰되는 구조와 통계적 검증 결과**를 근거로 내립니다.

이 결과는 이후 EmoNet의 내부 상태 표현을 개선하거나, 어떤 뉴런·연결·시간 구간이 발견된 군집을 만드는 데 기여하는지 추가 연구하는 기준으로 활용할 수 있습니다.

---

# 수행평가 평가 요소 대응표

| 수행평가 요구 요소 | 이 프로젝트에서의 계획 |
|---|---|
| 문제 정의 및 목적 | 감정 정보 없이 RAW TRACE의 안정적 군집을 찾고 감정과의 연관성을 사후 검증 |
| 데이터 수집 경로·방법·내용 | EmoNet에서 생성된 120개 RAW TRACE를 독립 저장소로 이전 |
| 결측치·이상치·통합·삭제 | JSON 통합, 비정상/저정보 특징 점검, leakage 변수 제거 |
| 새로운 열 생성 | 활성도, edge, E/I/M, K, stim, node frequency, temporal feature 생성 |
| 정규화 | log1p(K), Z-score 표준화 |
| 데이터 탐색 | 기술통계, feature 분포, 상관관계, PCA |
| 알고리즘 1 | K-means Clustering |
| 알고리즘 2 | Hierarchical Clustering |
| 알고리즘 선정 이유 | 서로 다른 군집 방식에서 잠재 구조가 재현되는지 확인 |
| 평가 지표 | Silhouette Score, 군집 안정성 |
| 추가 검증 | Cluster-Emotion 비교, Permutation Test, same/different emotion 거리 비교 |
| 모델 활용 | TRACE 내부에 감정과 관련된 잠재 구조가 있는지 데이터 기반으로 판단 |

---

# 데이터 및 현재 구현 상태

```text
isThereEmotion/
├─ data/
│  ├─ raw/                         # 120개의 원본 raw_trace.json (Git LFS)
│  ├─ processed/
│  │  └─ features.csv              # RAW TRACE에서 추출한 sample-level 특징
│  ├─ source_manifest.csv          # 원본 데이터 검증 정보
│  └─ SOURCE_COMMIT.txt
├─ scripts/
│  ├─ 01_build_features.py         # RAW → feature table
│  ├─ 02_eda.py                    # 기존 탐색 코드 — 군집 연구에 맞게 수정 예정
│  └─ 03_train.py                  # 기존 분류 실험 코드 — 본 연구의 핵심 분석은 아님
├─ results/
└─ requirements.txt
```

현재 `features.csv`는 RAW 120개에서 생성된 기본 특징 테이블입니다. 앞으로는 이 테이블을 그대로 믿고 사용하는 것이 아니라 **군집 분석에 필요한 특징만 다시 검토·선택**합니다.

다음 구현 순서:

```text
01_build_features.py
        ↓
02_blind_eda.py
        ↓
03_cluster_kmeans.py
        ↓
04_cluster_hierarchical.py
        ↓
05_cluster_stability.py
        ↓
06_emotion_validation.py
```

---

# 실행 환경

분석의 핵심 계산은 **pandas + NumPy**로 구현합니다.

```bash
git clone https://github.com/aether0xff-afk/isThereEmotion.git
cd isThereEmotion

git lfs pull

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
python scripts/01_build_features.py
```

---

# 해석 범위

이 프로젝트에서 다음과 같은 주장은 하지 않습니다.

> ❌ "군집이 있으므로 EmoNet은 실제 감정을 느낀다."

데이터가 충분히 뒷받침한다면 다음 수준까지 주장하는 것을 목표로 합니다.

> **감정 정보를 사용하지 않은 분석에서 EmoNet의 RAW 뉴런 TRACE가 안정적인 내부 상태 군집을 형성하고, 해당 군집이 외부 감정 범주와 무작위보다 강한 연관성을 보인다면, TRACE 내부에 감정과 체계적으로 관련된 잠재 구조가 존재한다는 증거로 해석할 수 있다.**
