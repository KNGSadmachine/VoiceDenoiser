import logging
from pathlib import Path

REPO_ID = "ResembleAI/resemble-enhance"
REPO_DIR = Path(__file__).parent.parent / "model_repo"

MODEL_FILES = [
    "enhancer_stage2/hparams.yaml",
    "enhancer_stage2/ds/G/latest",
    "enhancer_stage2/ds/G/default/mp_rank_00_model_states.pt",
]

logger = logging.getLogger(__name__)


def download() -> Path:
    run_dir = REPO_DIR / "enhancer_stage2"
    if all((REPO_DIR / f).is_file() for f in MODEL_FILES):
        return run_dir

    from huggingface_hub import hf_hub_download

    logger.info("Downloading the model...")
    for filename in MODEL_FILES:
        hf_hub_download(REPO_ID, filename, local_dir=REPO_DIR)
    return run_dir


if __name__ == "__main__":
    download()
