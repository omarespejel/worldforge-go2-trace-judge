# Media Review Recommendation

Recommended public set:
- `artifacts/showcase/worldforge_go2_trace_judge_showcase.mp4`
- `submission_bundle/demo_video.mp4` if needed as lightweight fallback
- curated Go2 stills under `artifacts/live_ciro/` after visual review
- real-photo-edit derived examples after approval

Recommended local-only / do not publish by default:
- `data/go2_camera_recording.mp4` raw full recording: keep as source evidence locally, but exclude from Hugging Face/public repo release unless reviewed.
- generated fake procedural images under `hf_dataset/images/synthetic/`: remove from the dataset direction.
- unapproved cut-paste previews under `hf_dataset/images/synthetic_cutpaste/`: keep only if replaced by approved real-photo-edit generator.

Deletion policy:
- Do not hard delete today. Move accidental/private footage to `private_media/` or exclude via `.gitignore`, then publish only curated derivatives.
