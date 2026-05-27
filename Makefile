PYTHON ?= python3
INPUT_VIDEO ?= data/go2_camera_recording.mp4
TARGET ?= green
UNSAFE_COLORS ?=
REPLAY_DIR ?= artifacts/replay_run
DATASET_JSONL ?= dataset/go2_trace_candidates.jsonl
RANKER_DIR ?= artifacts/ranker_smoke
AUDIT_DIR ?= artifacts/dataset_audit

.PHONY: check replay report review dataset audit ranker hf-dataset bundle package all clean-generated photo-smoke

all: check replay report review dataset audit ranker bundle package

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

hf-dataset:
	$(PYTHON) scripts/build_hf_decision_trace_dataset.py --clean --synthetic-count 180

bundle:
	$(PYTHON) scripts/build_submission_bundle.py

photo-smoke:
	$(PYTHON) scripts/go2_find_colored_target.py \
		--target green \
		--unsafe-colors red,yellow \
		--max-steps 1 \
		--dry-run-frame data/go2_camera_photo.jpg \
		--run-id photo-smoke \
		--output-dir artifacts/photo_smoke

package:
	cd .. && zip -qr worldforge-go2-trace-judge.zip worldforge-go2-trace-judge \
		-x 'worldforge-go2-trace-judge/.git/*'

clean-generated:
	rm -rf artifacts/replay_run/frames artifacts/replay_run/annotated_frames
