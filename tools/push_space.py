"""Create/update the Hugging Face Space from the local hf-space/ folder.

    set HF_TOKEN=hf_xxx        (or be logged in with `huggingface-cli login`)
    python tools/push_space.py

Reads no secrets from disk; the token comes from the HF_TOKEN environment
variable (or your cached HF login).
"""

from __future__ import annotations

import os

from huggingface_hub import HfApi, create_repo

REPO_ID = os.environ.get("SNFY_SPACE_ID", "naitik0409/sonify1026")


def main() -> None:
    create_repo(REPO_ID, repo_type="space", space_sdk="gradio", exist_ok=True)
    api = HfApi()
    api.upload_folder(
        folder_path="hf-space",
        repo_id=REPO_ID,
        repo_type="space",
        commit_message="Sonify 1026 demo",
    )
    print(f"Pushed -> https://huggingface.co/spaces/{REPO_ID}")


if __name__ == "__main__":
    main()
