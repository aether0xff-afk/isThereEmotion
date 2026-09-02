# isThereEmotion

> TRACE에 감정이 있을까요?

EmoNet이 감정 대화 입력을 처리할 때 남긴 **뉴런 활성 TRACE만으로 원 데이터의 감정 범주를 구분할 수 있는지** 확인하는 독립 데이터 분석 프로젝트입니다.

## 연구 질문

**감정 대화 입력에 대해 생성된 EmoNet 뉴런 TRACE의 시간적·구조적 특징만을 이용해 원 데이터의 감정 범주를 분류할 수 있는가? 감정 범주별 내부 활성 양상에는 어떤 차이가 있는가?**

핵심은 감정 라벨을 직접 예측하기 위해 텍스트나 EmoNet의 후단 감정 출력값을 사용하는 것이 아니라, 실제 뉴런 활성 과정 자체에 구분 가능한 패턴이 남는지를 보는 것입니다.

## 데이터

원본 데이터는 이 저장소의 `data/raw/<sample_id>/raw_trace.json`에 포함합니다. 파일 크기가 커서 Git LFS로 관리합니다.

원본 출처는 EmoNet의 `v4/outputs/research/trajectory_batch_matrix120_v1`이며, 복사 후 분석은 이 저장소만으로 수행할 수 있습니다.

각 TRACE에는 tick별로 다음 정보가 있습니다.

- 활성 뉴런: `active_nodes`
- 발화 연결: `edges_fired`
- 뉴런 종류: excitatory / inhibitory / modulatory
- 뉴런 상태값 `K`
- 자극 벡터 `stim_vec`

정답은 원 데이터의 `input_meta.label`을 사용합니다.

### 데이터 누수 방지

모델 입력 X에는 **tick 기반 TRACE에서 직접 계산한 특징만** 사용합니다.

다음과 같은 후단 출력·감정 해석 결과는 입력 특징으로 사용하지 않습니다.

- latent `z_*`
- 신경전달물질 모사 값
- style / predicted style 값
- `top_emotions`
- `dominant_global_signal`
- 생성된 LLM response

또한 같은 대화가 train/test에 섞이는 것을 줄이기 위해 가능하면 `talk_id` 기준 group split을 사용합니다.

## 전처리 및 특징 생성

중첩된 raw JSON을 샘플 단위 표 데이터로 바꿉니다.

주요 특징 예시:

- 평균/표준편차/최대 활성 뉴런 수
- 초기 10 tick 활성 증가 기울기
- early/late 활성 평균
- 최대 활성의 50%, 90%에 도달한 시점
- unique active node 수와 비율
- 평균/표준편차/최대 fired edge 수
- unique edge 수와 edge 재사용 비율
- 연속 tick 간 active-set Jaccard similarity
- E/I/M 뉴런 비율
- neuron type별 K 평균/표준편차
- neuron type별 stimulus norm 평균/표준편차
- 각 뉴런의 전체 tick 대비 활성 빈도 `node_freq_***`

```bash
python scripts/01_build_features.py
```

결과는 `data/processed/features.csv`에 저장됩니다.

## 탐색적 분석

```bash
python scripts/02_eda.py
```

생성 항목:

- 감정 label별 표본 수
- 수치형 특징 요약
- label별 주요 TRACE 특징 평균
- TRACE 특징 PCA 2차원 투영

## 모델

두 가지 서로 다른 성격의 분류기를 비교합니다.

1. **Logistic Regression**: 선형 기준선
2. **Random Forest**: 비선형 특징 상호작용을 확인하는 모델

```bash
python scripts/03_train.py
```

평가 지표:

- Accuracy
- Balanced Accuracy
- Macro F1
- Weighted F1
- Confusion Matrix

불균형 감정 범주가 있을 수 있으므로 최종 비교에서는 특히 **Macro F1**을 중요하게 봅니다.

## 실행

```bash
git clone https://github.com/aether0xff-afk/isThereEmotion.git
cd isThereEmotion
git lfs pull
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python scripts/01_build_features.py
python scripts/02_eda.py
python scripts/03_train.py
```

## 구조

```text
isThereEmotion/
├─ data/
│  ├─ raw/                 # 원본 raw_trace.json
│  └─ processed/           # 추출된 features.csv
├─ scripts/
│  ├─ 01_build_features.py
│  ├─ 02_eda.py
│  └─ 03_train.py
├─ results/                # EDA / 모델 결과
└─ requirements.txt
```

## 현재 해석 범위

이 프로젝트가 확인하려는 것은 **TRACE가 감정 범주와 통계적으로 구분되는 내부 패턴을 가지는가**입니다. 분류 성능이 높더라도 그것만으로 모델이 실제 감정을 느낀다고 주장하지 않습니다.
