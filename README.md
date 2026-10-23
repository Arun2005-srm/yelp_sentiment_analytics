# Tonewise — Customer sentiment intelligence

A React dashboard connected to a FastAPI inference service, with local SQLite history and your original Spark project included.

## Start the dashboard

Double-click **Start Tonewise.cmd**. The included production frontend is served with the API at http://127.0.0.1:8000. First startup may take a few seconds to load DistilBERT onto the GPU (or CPU if CUDA is unavailable).

If Python dependencies are missing, run from this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

Then run the launcher again. The launcher prefers `.venv`, then the local `.runtime` helper, then installed Python packages. Python 3.10+ and Node.js 18+ are recommended. Node is needed for editing/rebuilding, not for the included compiled dashboard.

For frontend development, run `npm ci`, then `npm run dev` in one terminal, and `python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000` in another. Vite proxies `/api` to the backend. Rebuild with `npm run build`; restart the backend after rebuilding. API docs: http://127.0.0.1:8000/docs.

## Included features

- Single review analysis with all three sentiment probabilities, confidence, token count, truncation notice, and inference time.
- Batch CSV import: 1–200 reviews with a `text`, `review`, `comment`, or `feedback` column; inference uses groups of eight.
- Persistent SQLite history, search, sentiment filters, low-confidence filter, review details, sorting, and pagination.
- Date filters, daily sentiment charts, sentiment mix, and seven KPIs: net sentiment, feedback volume, positive share, negative share, average confidence, reviews below 80% confidence, and average inference time.
- Filtered CSV export and full-history API export; spreadsheet formulas are escaped.
- Explicit demo mode with separate illustrative sample data. No demo review is written to the real history database.
- Model status, reload control, original evaluation metrics, and a Spark workflow overview.
- Responsive layout, keyboard focus indicators, accessible input labels, and reduced-motion support.

Net sentiment = (positive count − negative count) / total × 100. Confidence is the highest softmax probability, not measured accuracy. Review recommended means confidence < 80%; it is a triage heuristic. Inference time is the processing time per review for its request; it excludes network and database time. Dates and trend buckets use UTC; review details show your computer's local time. KPIs and exports follow the selected period. Text search and sentiment filters narrow the feedback list and exported report.

## Model provenance

**Active weights:** `models/yelp-distilbert/model.safetensors`, downloaded from [FinchW/my-yelp-sentiment-model-finetuned](https://huggingface.co/FinchW/my-yelp-sentiment-model-finetuned). The model card specifies 0=negative, 1=neutral, 2=positive. Its card reports evaluation accuracy 84.846% and F1 0.84949. These are publisher-reported results, not independently reproduced here.

The tokenizer is loaded from the copied `models/distilbert` folder. Both models use the DistilBERT uncased vocabulary. The API performs inference entirely locally and uses at most the first 512 tokens. It never substitutes heuristic predictions or downloads weights at startup.

The original project's reported 92.35% test accuracy and 0.9207 weighted F1 belong to the original project, **not** the downloaded replacement weights. The Model & pipeline screen labels these as the original supplied evaluation. Original checkpoints contain scheduler/scaler/trainer state but no learned weight file; all those supplied files are preserved.

To download again: `python scripts/download_model.py` (internet required). Set `MODEL_DIR` to use another local model folder with a documented three-class label mapping. Generic LABEL_0/1/2 are accepted only for the included FinchW folder. A custom model must provide explicit negative/neutral/positive (or Negatif/Netral/Positif) labels.

## Project layout

- `src/` — React UI and styles
- `backend/main.py` — FastAPI, local model loader, SQLite persistence
- `dist/` — ready-to-run production frontend
- `models/yelp-distilbert/` — downloaded active model
- `models/distilbert/` — all supplied tokenizer/configuration/checkpoint files
- `notebooks/` — all five original Spark/training/analysis notebooks
- `outputs/` — all original figures, metrics, and prediction cache
- `README.pipeline.md` — unmodified original project README
- `requirements-spark.txt` — optional notebook dependencies; Spark is not needed for live inference
- `tests/` — API validation, persistence, batch, export, model-missing, and real inference tests

The supplied notebooks retain their original dataset paths. Raw Yelp data and processed Parquet were not present in the source folder. To rerun training, supply the data and adapt those notebook paths to your environment; consult `README.pipeline.md`. The online dashboard does not rerun Spark for each review.

## Verification

```powershell
python -m pytest tests -q
npm run build
```

Tests use an isolated temporary database. Real-model integration tests load the downloaded weights and check positive/negative predictions, batch persistence, and the 512-token truncation limit. User-facing history is not seeded by tests.

The server binds to loopback for use on this computer. SQLite history lives at `data/history.sqlite3`; override `DATABASE_PATH` if needed. This local app has no multi-user authentication. Review model confidence carefully for ambiguous, sarcastic, multilingual, and out-of-domain feedback.
