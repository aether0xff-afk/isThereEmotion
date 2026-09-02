# isThereEmotion

> **TRACE에 감정이라고 부를 만한 구조가 있을까요?**

EmoNet이 감정 대화 입력을 처리할 때 남긴 **RAW 뉴런 TRACE 자체를 분석하여**, 감정 정보를 보지 않은 상태에서도 안정적인 내부 상태 군집이 형성되는지 확인하는 수행평가용 데이터 분석 프로젝트입니다.

분석 코드는 여러 파일로 나누지 않고 **`isThereEmotion.ipynb` 하나에 모두 포함**되어 있습니다.

---

## 1. 연구 질문

> **EmoNet의 RAW 뉴런 TRACE를 감정 정보 없이 분석했을 때 안정적인 내부 상태 군집이 발견되는가? 또한 발견된 군집은 외부 감정 범주와 우연 이상의 연관성을 가지는가?**

목표는 감정 label을 잘 맞히는 분류기를 만드는 것이 아닙니다.

```text
RAW TRACE
    ↓
감정 label을 사용하지 않은 특징 추출
    ↓
PCA + 비지도 군집
    ↓
군집 품질 및 안정성 검증
    ↓
────────────────────
여기까지 감정 label 봉인
────────────────────
    ↓
감정 label 공개
    ↓
군집 ↔ 감정 연관성 검증
    ↓
Permutation test로 우연 여부 확인
```

즉, **감정이라는 정답을 먼저 보고 군집을 만드는 것이 아니라 TRACE에서 독립적으로 발견된 구조가 사후적으로 감정과 연결되는지**를 검증합니다.

---

## 2. 데이터

기존 EmoNet 실험의 `trajectory_batch_matrix120_v1` RAW TRACE를 이 저장소로 독립 이전했습니다.

- sample 수: **120개**
- 형식: JSON
- 위치: `data/raw/<sample_id>/raw_trace.json`
- 전체 RAW 크기: 약 **1.81 GB**
- Git LFS 사용
- `data/source_manifest.csv`에 sample ID, label, 원본 크기, SHA-256, 경로 기록

주요 RAW 항목:

| 항목 | 의미 |
|---|---|
| `active_nodes` | tick별 활성 뉴런 |
| `edges_fired` | 발화된 연결 |
| `node_id` | 뉴런 식별자 |
| `neuron_type` | excitatory / inhibitory / modulatory |
| `K` | 뉴런 내부 상태값 |
| `stim_vec` | 4차원 자극 벡터 |
| tick 순서 | 내부 상태의 시간적 변화 |

---

## 3. 데이터 누수 방지

군집 생성 단계에서는 `input_meta.label`을 사용하지 않습니다.

또한 다음 값도 feature에서 제외합니다.

```text
latent z_*
dopamine / serotonin / ...
style s_*
predicted style s_hat_*
top_emotions
dominant_global_signal
LLM response
```

감정 label은 군집 수와 군집 구조를 확정한 뒤 **사후 검증 단계에서만** 공개합니다.

---

## 4. 전처리

노트북에서 RAW JSON을 sample-level feature table로 변환합니다.

주요 feature:

- 평균/표준편차/최대 활성 뉴런 수
- 초기/후기 활성도와 활성 변화 기울기
- unique active node 수와 비율
- 뉴런별 활성 빈도 `node_freq_000 ~ node_freq_255`
- fired edge 평균/표준편차/재사용률
- 연속 tick active-set Jaccard similarity
- excitatory / inhibitory / modulatory 비율
- `K` 평균/표준편차/시간 기울기
- `stim_vec` 각 축의 평균/표준편차/시간 기울기

추가 처리:

- 비정상 TRACE 제거
- `K' = sign(K) × log(1 + |K|)` 변환
- 결측값 중앙값 대치
- 저분산 feature 제거
- Z-score 표준화
- NumPy SVD 기반 PCA

---

## 5. 탐색 및 시각화

**분석 계산은 NumPy와 pandas만 사용하며, matplotlib은 계산된 결과를 그림으로 표시하는 용도로만 사용합니다.**

노트북에는 다음 시각화가 포함됩니다.

1. `ticks`, 활성 비율, fired edge, `logK`의 분포 histogram
2. PCA 누적 설명분산 그래프
3. 감정 label을 사용하지 않은 PCA 2차원 산점도
4. `k=2~10`에 대한 K-means Silhouette score 그래프
5. PCA 공간의 K-means 군집 산점도
6. PCA 공간의 Hierarchical 군집 산점도
7. seed 변화에 따른 군집 안정성 ARI 그래프
8. feature shuffle negative-control Silhouette 분포
9. 감정 label permutation NMI 분포와 실제 NMI 비교
10. 같은 감정 vs 다른 감정 sample 간 평균 거리 비교
11. 거리 차이에 대한 permutation null 분포

---

## 6. 알고리즘

### K-means Clustering

TRACE 특징이 거리 공간에서 자연스럽게 몇 개의 중심 주변으로 모이는지 확인합니다.

- NumPy 직접 구현
- `k = 2 ~ 10` 비교
- 여러 초기화 결과 중 inertia 최소 결과 사용

### Average-Linkage Hierarchical Clustering

K-means와 다른 원리에서도 유사한 군집 구조가 나타나는지 확인합니다.

- pairwise distance 기반
- average linkage 직접 구현
- K-means와의 구조 일치도를 ARI로 비교

---

## 7. 평가 지표

### Silhouette Score

군집 내부 응집도와 군집 사이 분리도를 평가합니다.

### Adjusted Rand Index (ARI)

seed를 바꾸거나 군집 알고리즘을 바꾸어도 비슷한 구조가 유지되는지 확인합니다.

### Negative Control

feature를 sample별로 독립 shuffle하여 내부 공동 구조를 깨뜨린 뒤 실제 데이터의 Silhouette와 비교합니다.

---

## 8. 감정과의 사후 검증

군집 구조를 먼저 고정한 뒤 감정 label을 공개합니다.

### NMI + Permutation Test

실제 cluster와 감정 label의 NMI를 구하고, label을 무작위로 2,000번 섞은 null distribution과 비교합니다.

### Same-emotion vs Different-emotion Distance

군집 알고리즘과 독립적으로,

```text
같은 감정 sample 간 평균 TRACE 거리
vs
다른 감정 sample 간 평균 TRACE 거리
```

를 비교하고 permutation test를 수행합니다.

---

## 9. 결과 해석 원칙

| 결과 | 해석 |
|---|---|
| 뚜렷한 군집 없음 | 현재 TRACE에서 안정적인 잠재 상태 구조의 증거가 부족함 |
| 군집 있음 + 감정과 관계 없음 | 내부 구조는 있지만 감정 구조라고 보기는 어려움 |
| 안정적 군집 + 감정과 우연 이상의 연관성 | TRACE 내부에 감정과 체계적으로 연관된 잠재 구조가 있다는 증거 |

결과가 좋더라도 **EmoNet이 실제 감정을 느낀다고 주장하지 않습니다.**

---

## 10. 실행

```bash
git clone https://github.com/aether0xff-afk/isThereEmotion.git
cd isThereEmotion
git lfs pull
pip install -r requirements.txt
jupyter notebook isThereEmotion.ipynb
```

`requirements.txt`:

```text
numpy
pandas
matplotlib
```

---

## 구조

```text
isThereEmotion/
├─ isThereEmotion.ipynb      # 전체 분석 + 시각화
├─ data/
│  ├─ raw/                   # RAW TRACE 120개
│  ├─ source_manifest.csv
│  └─ SOURCE_COMMIT.txt
├─ requirements.txt
├─ README.md
└─ LICENSE
```

## 핵심 한 문장

> **감정 label을 보지 않고 발견한 안정적인 RAW TRACE 군집이, label을 공개한 뒤에도 우연 이상의 감정 연관성을 보이는지를 검증한다.**
