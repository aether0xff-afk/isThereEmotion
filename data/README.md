# Data

`data/raw/`에는 EmoNet TRACE 원본 JSON을 저장한다. 원본 파일은 Git LFS로 관리하며, 저장소를 clone한 뒤 `git lfs pull`을 실행하면 분석에 필요한 raw JSON 전체를 받을 수 있다.

분석 입력 X는 raw JSON의 `ticks` 내부 정보만 사용한다: `active_nodes`, `edges_fired`, `node_states.neuron_type`, `node_states.K`, `node_states.stim_vec`.

정답 y는 `input_meta.label`을 사용한다. `sample_id`, `talk_id`, `persona_id`, `profile_id`는 식별 및 데이터 분할에만 사용하고 모델 입력에서는 제외한다.

후단 출력이나 감정 판단 결과를 직접 담은 변수는 모델 입력에서 제외해 데이터 누수를 막는다.

`scripts/01_build_features.py`를 실행하면 `data/processed/features.csv`가 생성된다.
