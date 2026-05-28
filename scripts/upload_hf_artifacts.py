from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


def upload_folder(api: HfApi, repo_id: str, repo_type: str, folder: Path, message: str) -> str:
    if not folder.exists():
        raise RuntimeError(f"Missing folder: {folder}")
    api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type=repo_type,
        folder_path=str(folder),
        commit_message=message,
    )
    return f"https://huggingface.co/{'datasets/' if repo_type == 'dataset' else ''}{repo_id}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload HF-ready dataset/model artifacts.")
    parser.add_argument("--dataset-repo", default="espejelomar/worldforge-go2-dimos-replay-world-pairs")
    parser.add_argument("--model-repo", default="espejelomar/go2-dimos-replay-latent-dynamics")
    parser.add_argument("--dataset-dir", default="hf_dataset_dimos_replay")
    parser.add_argument("--model-dir", default="hf_model_dimos_replay_latent")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("Set HF_TOKEN in the environment; do not commit it.")

    api = HfApi(token=token)
    dataset_url = upload_folder(
        api,
        args.dataset_repo,
        "dataset",
        Path(args.dataset_dir),
        "Add DimOS Go2 replay world-model pairs",
    )
    model_url = upload_folder(
        api,
        args.model_repo,
        "model",
        Path(args.model_dir),
        "Add Go2 DimOS replay latent dynamics head",
    )
    print(f"dataset_url={dataset_url}")
    print(f"model_url={model_url}")


if __name__ == "__main__":
    main()
