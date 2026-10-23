"""Run this from any directory to restore the active model locally."""
import os
from pathlib import Path
root = Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME', str(root / '.cache/huggingface'))
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
from huggingface_hub import snapshot_download
snapshot_download(repo_id='FinchW/my-yelp-sentiment-model-finetuned', local_dir=root/'models/yelp-distilbert')
print('Model downloaded. Start Tonewise or click Reload model.')
