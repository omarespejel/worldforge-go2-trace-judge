PYTHON ?= python3
INPUT_VIDEO ?= data/go2_camera_recording.mp4
TARGET ?= red
UNSAFE_COLORS ?= green,yellow
REPLAY_DIR ?= artifacts/replay_run
DATASET_JSONL ?= dataset/go2_trace_candidates.jsonl
RANKER_DIR ?= artifacts/ranker_smoke
AUDIT_DIR ?= artifacts/dataset_audit
REAL_PHOTO_EDIT_COUNT ?= 480
MICRO_WORLD_MODEL ?= artifacts/micro_world_scorer/model.json
MICRO_JEPA_MODEL ?= artifacts/micro_jepa_scorer/model.json
MICRO_WORLD_IMAGE ?= artifacts/live_ciro/direct_camera_unsafe_path.jpg
DIMOS_REPLAY_DATASET ?= hf_dataset_dimos_replay
DIMOS_REPLAY_LATENT_MODEL ?= artifacts/dimos_replay_latent_dynamics
DIMOS_REPLAY_SOURCE_DATASETS ?= go2_short,markers_go2,go2_bigoffice,go2_hongkong_office,go2_slamabuse1,go2_slamabuse2,go2_china_office
DIMOS_REPLAY_MAX_PAIRS_PER_SOURCE ?= 500

.PHONY: check replay report review dataset audit ranker real-photo-edit hf-dataset micro-world-scorer micro-world-demo micro-jepa-scorer micro-jepa-demo jepa-stretch model-honesty-audit dinov2-scorer dimos-replay-dataset dimos-replay-latent-dynamics replay-mpc-demo dimos-replay-stretch dimos-sim-probe ml-stretch final-video bundle package all hackathon-final clean-generated photo-smoke

all: hackathon-final

hackathon-final: check real-photo-edit hf-dataset micro-world-scorer micro-world-demo final-video bundle package

check:
	$(PYTHON) -m py_compile scripts/*.py

replay:
	$(PYTHON) scripts/go2_trace_replay.py \
		--input-video "$(INPUT_VIDEO)" \
		--output-dir "$(REPLAY_DIR)" \
		--run-id go2-camera-replay \
		--target "$(TARGET)" \
		--unsafe-colors "$(UNSAFE_COLORS)" \
		--fps 2 \
		--width 960

report:
	$(PYTHON) scripts/build_replay_report.py --replay-dir "$(REPLAY_DIR)"

review:
	$(PYTHON) scripts/build_human_review_pack.py --replay-dir "$(REPLAY_DIR)"

dataset:
	$(PYTHON) scripts/collect_trace_dataset.py \
		--trace-dir "$(REPLAY_DIR)/trace" \
		--output-jsonl "$(DATASET_JSONL)" \
		--summary-output dataset/go2_trace_dataset_summary.json

audit:
	$(PYTHON) scripts/audit_trace_dataset.py \
		--replay-dir "$(REPLAY_DIR)" \
		--dataset-jsonl "$(DATASET_JSONL)" \
		--output-dir "$(AUDIT_DIR)"

ranker:
	$(PYTHON) scripts/train_tiny_ranker.py \
		--dataset-jsonl "$(DATASET_JSONL)" \
		--output-dir "$(RANKER_DIR)"

real-photo-edit:
	$(PYTHON) scripts/build_real_photo_edit_dataset.py \
		--count "$(REAL_PHOTO_EDIT_COUNT)" \
		--clean

hf-dataset:
	$(PYTHON) scripts/build_hf_decision_trace_dataset.py --clean

micro-world-scorer:
	$(PYTHON) scripts/train_micro_world_scorer.py \
		--dataset-dir hf_dataset \
		--output-dir artifacts/micro_world_scorer
	mkdir -p hf_model
	cp artifacts/micro_world_scorer/model.json hf_model/model.json
	cp artifacts/micro_world_scorer/eval_report.json hf_model/eval_report.json
	cp artifacts/micro_world_scorer/predictions_sample.json hf_model/predictions_sample.json

micro-world-demo:
	$(PYTHON) scripts/run_micro_world_scorer_demo.py \
		--image "$(MICRO_WORLD_IMAGE)" \
		--model "$(MICRO_WORLD_MODEL)" \
		--run-id latest \
		--clean

micro-jepa-scorer:
	$(PYTHON) scripts/train_micro_jepa_scorer.py \
		--dataset-dir hf_dataset \
		--output-dir artifacts/micro_jepa_scorer
	mkdir -p hf_model_jepa
	cp artifacts/micro_jepa_scorer/model.json hf_model_jepa/model.json
	cp artifacts/micro_jepa_scorer/eval_report.json hf_model_jepa/eval_report.json
	cp artifacts/micro_jepa_scorer/predictions_sample.json hf_model_jepa/predictions_sample.json

micro-jepa-demo:
	$(PYTHON) scripts/run_micro_world_scorer_demo.py \
		--image "$(MICRO_WORLD_IMAGE)" \
		--model "$(MICRO_JEPA_MODEL)" \
		--output-dir artifacts/micro_jepa_demo \
		--run-id latest \
		--clean

jepa-stretch: check micro-jepa-scorer micro-jepa-demo

model-honesty-audit:
	$(PYTHON) scripts/audit_model_honesty.py \
		--dataset-dir hf_dataset \
		--output-dir artifacts/model_audit

dinov2-scorer:
	$(PYTHON) scripts/train_dinov2_scorer.py \
		--dataset-dir hf_dataset \
		--output-dir artifacts/dinov2_scorer \
		--alpha 0.01
	mkdir -p hf_model_dinov2
	cp artifacts/dinov2_scorer/model.json hf_model_dinov2/model.json
	cp artifacts/dinov2_scorer/eval_report.json hf_model_dinov2/eval_report.json
	cp artifacts/dinov2_scorer/predictions_sample.json hf_model_dinov2/predictions_sample.json

ml-stretch: check jepa-stretch model-honesty-audit dinov2-scorer

dimos-replay-dataset:
	$(PYTHON) scripts/build_dimos_replay_world_dataset.py \
		--datasets "$(DIMOS_REPLAY_SOURCE_DATASETS)" \
		--output-dir "$(DIMOS_REPLAY_DATASET)" \
		--max-pairs-per-source "$(DIMOS_REPLAY_MAX_PAIRS_PER_SOURCE)" \
		--clean

dimos-replay-latent-dynamics:
	$(PYTHON) scripts/train_dimos_replay_latent_dynamics.py \
		--dataset-dir "$(DIMOS_REPLAY_DATASET)" \
		--output-dir "$(DIMOS_REPLAY_LATENT_MODEL)" \
		--alpha 1000
	mkdir -p hf_model_dimos_replay_latent
	cp "$(DIMOS_REPLAY_LATENT_MODEL)/model.json" hf_model_dimos_replay_latent/model.json
	cp "$(DIMOS_REPLAY_LATENT_MODEL)/eval_report.json" hf_model_dimos_replay_latent/eval_report.json
	cp "$(DIMOS_REPLAY_LATENT_MODEL)/candidate_scores_sample.json" hf_model_dimos_replay_latent/candidate_scores_sample.json
	cp "$(DIMOS_REPLAY_LATENT_MODEL)/README.md" hf_model_dimos_replay_latent/README.md

replay-mpc-demo:
	$(PYTHON) scripts/run_replay_mpc_demo.py \
		--dataset-dir "$(DIMOS_REPLAY_DATASET)" \
		--model-dir hf_model_dimos_replay_latent \
		--output-dir artifacts/replay_mpc_demo \
		--clean

dimos-replay-stretch: check dimos-replay-dataset dimos-replay-latent-dynamics replay-mpc-demo

dimos-sim-probe:
	$(PYTHON) scripts/dimos_simulation_probe.py

final-video:
	$(PYTHON) scripts/build_final_showcase_video.py

bundle:
	$(PYTHON) scripts/build_submission_bundle.py

photo-smoke:
	$(PYTHON) scripts/go2_find_colored_target.py \
		--target "$(TARGET)" \
		--unsafe-colors "$(UNSAFE_COLORS)" \
		--max-steps 1 \
		--dry-run-frame artifacts/live_ciro/direct_camera_unsafe_path.jpg \
		--run-id photo-smoke \
		--output-dir artifacts/photo_smoke

package:
	git archive \
		--format=zip \
		--prefix=worldforge-go2-trace-judge/ \
		--output=../worldforge-go2-trace-judge.zip \
		HEAD

clean-generated:
	rm -rf artifacts/replay_run/frames artifacts/replay_run/annotated_frames
	rm -rf artifacts/showcase/frames artifacts/showcase/final_frames artifacts/showcase/robot_video_frames
	rm -rf artifacts/micro_world_demo/*/video_frames
