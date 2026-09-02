# Data

`data/raw/`에는 EmoNet 실험에서 생성된 **120개의 원본 `raw_trace.json`**이 저장되어 있습니다. 원본은 약 1.81 GB이며 Git LFS로 관리합니다.

저장소를 clone한 뒤 아래 명령으로 실제 RAW 파일을 받습니다.

```bash
git lfs pull
```

각 파일의 경로, sample ID, 원 감정 label, 파일 크기, SHA-256은 `source_manifest.csv`에 기록되어 있고, 원 데이터 출처 commit은 `SOURCE_COMMIT.txt`에 남겨 두었습니다.

## 분석에 사용하는 RAW 정보

군집 feature는 `ticks` 내부의 다음 값에서 직접 생성합니다.

- `active_nodes`
- `edges_fired`
- `node_states.node_id`
- `node_states.neuron_type`
- `node_states.K`
- `node_states.stim_vec`
- tick 순서

`input_meta.label`은 **군집 생성에는 사용하지 않고**, 군집 구조와 군집 수를 먼저 확정한 뒤 사후 검증에서만 사용합니다.

latent, neurotransmitter-like outputs, style 값, `top_emotions`, `dominant_global_signal`, LLM response처럼 후단 감정 해석과 직접 관련된 변수는 군집 feature에서 제외합니다.

## 전처리 코드

별도의 `scripts/` 폴더는 사용하지 않습니다.

모든 RAW 파싱, feature 생성, 품질 확인, 표준화, PCA, 군집 분석 및 감정 사후 검증은 저장소 루트의 다음 노트북에 들어 있습니다.

```text
isThereEmotion.ipynb
```

노트북을 실행하면 필요에 따라 다음 폴더가 생성됩니다.

```text
data/processed/
├─ trace_features.csv
└─ metadata_labels.csv

results/
├─ blind_cluster_comparison.csv
├─ cluster_stability.csv
├─ cluster_assignments.csv
└─ validation_summary.csv
```
