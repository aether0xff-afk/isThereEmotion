import json
from pathlib import Path


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip() + "\n"}


def code_from(path):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": Path(path).read_text(encoding="utf-8"),
    }


cells = [
    md("""
# isThereEmotion — RAW TRACE에서 감정 관련 잠재 군집 찾기

> **연구 질문**  
> EmoNet의 RAW 뉴런 TRACE를 감정 정보 없이 분석했을 때 안정적인 내부 상태 군집이 발견되는가?  
> 또한 발견된 군집은 외부 감정 범주와 우연 이상의 연관성을 가지는가?

이 노트북 하나에서 **RAW JSON 전처리 → 라벨 블라인드 EDA → PCA → K-means → 계층적 군집화 → 군집 안정성 → 감정 라벨 사후 검증 → permutation test**까지 수행한다.

분석 라이브러리는 **NumPy와 pandas만 사용**한다. `json`, `pathlib`은 Python 표준 라이브러리이다.

## 분석 원칙

1. 군집 생성과 군집 수 선택에는 감정 `label`을 사용하지 않는다.
2. `top_emotions`, `dominant_global_signal`, latent `z_*`, 신경전달물질 모사값, style/LLM response 등 후단 감정 해석 결과는 feature에서 제외한다.
3. 군집 품질과 안정성을 먼저 확정한 뒤에만 감정 label을 공개한다.
4. 최종 결론은 “모델이 감정을 느낀다”가 아니라 **TRACE 내부에 감정과 체계적으로 연관된 잠재 구조가 존재하는지**에 한정한다.
"""),
    code_from("notebook_parts/00_setup.py"),
    md("""
## 1. RAW TRACE 전처리 및 탐색

각 `raw_trace.json`은 여러 tick과 뉴런 상태가 중첩된 계층형 데이터이다. 이를 sample-level 수치 특징으로 바꾼다.

사용하는 RAW 정보는 `active_nodes`, `edges_fired`, `node_states.neuron_type`, `K`, `stim_vec`, tick 순서이다. 감정 label은 별도 metadata에 보관하고 **feature table과 물리적으로 분리**한다.

전처리에는 JSON 통합, 실패 TRACE 제거, 새로운 열 생성, `K`의 log 변환, 결측치 중앙값 처리, 저분산 열 제거, Z-score 표준화, PCA가 포함된다. 외부 plotting library를 사용할 수 없으므로 주요 분포는 NumPy 기반 ASCII histogram과 표로 확인한다.
"""),
    code_from("notebook_parts/01_preprocess.py"),
    md("""
## 2. 감정 label을 보지 않는 군집 분석

서로 다른 원리의 두 알고리즘을 비교한다.

- **K-means clustering**: 중심점과의 거리 기반
- **Average-linkage hierarchical clustering**: 가까운 sample/cluster를 순차적으로 병합

`k=2~10`을 모두 검사하고, 감정 label이 아니라 **Silhouette score**만으로 군집 수를 비교한다. 두 알고리즘의 평균 Silhouette가 가장 높은 `K*`를 고정한 뒤, ARI로 알고리즘 간 재현성과 seed/작은 perturbation에 대한 안정성을 확인한다. feature-wise shuffle negative control도 함께 수행한다.
"""),
    code_from("notebook_parts/02_cluster.py"),
    md("""
## 3. 🔓 감정 label 봉인 해제 — 사후 검증

**여기까지의 PCA, K*, 군집 결과는 감정 label을 사용하지 않고 먼저 확정한다.**

이제 처음으로 외부 감정 label을 공개한다.

- cluster ↔ emotion 연관성: **Normalized Mutual Information (NMI)**
- 우연 여부: **label permutation test**
- 특정 clustering 알고리즘에 의존하지 않는 보강 증거: **same-emotion distance vs different-emotion distance + permutation test**

현재 E-code는 세부 범주가 많고 희소한 class가 있으므로 NMI 절대값만 해석하지 않고, 같은 label 빈도를 유지한 permutation null distribution과 비교한다.
"""),
    code_from("notebook_parts/03_validate.py"),
    md("""
## 4. 결과 해석 기준

이 프로젝트에서는 하나의 숫자만으로 “감정이 있다”고 결론내리지 않는다.

- **뚜렷한 군집 자체가 없음** → 현재 TRACE에서 안정적인 잠재 상태 구조의 증거가 부족하다.
- **안정적인 군집은 있으나 emotion permutation test가 유의하지 않음** → 내부 구조는 있으나 감정 구조라고 부를 증거는 부족하다.
- **군집이 안정적이고 서로 다른 알고리즘에서도 재현되며 emotion과의 연관성이 permutation null보다 강함** → TRACE 내부에 **감정과 체계적으로 연관된 잠재 구조가 존재한다는 증거**가 된다.
- **same-emotion distance까지 더 작고 permutation test에서도 유의함** → 특정 clustering 알고리즘에만 의존하지 않는 독립적인 보강 증거가 된다.

어떤 경우에도 이 결과만으로 EmoNet이 주관적으로 감정을 “느낀다”고 주장하지 않는다.

## 수행평가 체크리스트

| 평가 항목 | 노트북에서 수행하는 내용 |
|---|---|
| 문제 정의 | RAW TRACE에서 감정 관련 잠재 군집이 발견되는가? |
| 데이터 수집 | EmoNet에서 생성한 독립 RAW TRACE 120개 |
| 전처리 | JSON 통합, 실패 TRACE 제거, 새 feature 생성, log 변환, 결측 처리, 저분산 제거, 표준화 |
| 탐색 | feature 분포, PCA 설명분산 |
| 알고리즘 1 | K-means clustering |
| 알고리즘 2 | Average-linkage hierarchical clustering |
| 선정 이유 | 서로 다른 원리에서도 내부 구조가 재현되는지 확인 |
| 평가 지표 | Silhouette score, ARI 기반 안정성 |
| 추가 검증 | NMI + permutation test, same/different emotion distance test |
| 활용 | TRACE 내부에 감정과 관련된 잠재 구조가 존재하는지 데이터로 판단 |

### 핵심 한 문장

**감정 label을 보지 않고 발견한 안정적인 TRACE 군집이, label을 공개한 뒤에도 우연 이상의 감정 연관성을 보이는지를 검증한다.**
"""),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

Path("isThereEmotion.ipynb").write_text(
    json.dumps(notebook, ensure_ascii=False, indent=1),
    encoding="utf-8",
)

# 최소 검증: JSON 재로딩 + 모든 코드 cell 문법 확인
loaded = json.loads(Path("isThereEmotion.ipynb").read_text(encoding="utf-8"))
for i, cell in enumerate(loaded["cells"]):
    if cell["cell_type"] == "code":
        compile(cell["source"], f"cell_{i}", "exec")

print("built isThereEmotion.ipynb")
print("cells:", len(loaded["cells"]))
