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
MICRO_WORLD_IMAGE ?= artifacts/live_ciro/direct_camera_unsafe_path.jpg

.PHONY: check replay report review dataset audit ranker real-photo-edit hf-dataset micro-world-scorer micro-world-demo final-video bundle package all hackathon-final clean-generated photo-smoke

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
