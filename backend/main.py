import csv
import io
import json
import os
import sqlite3
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.environ.get('MODEL_DIR', ROOT / 'models/yelp-distilbert'))
DB_PATH = Path(os.environ.get('DATABASE_PATH', ROOT / 'data/history.sqlite3'))
LABELS = ['negative', 'neutral', 'positive']
lock = threading.Lock()
state = {'status': 'loading', 'detail': 'Checking local model files', 'device': 'cpu'}
model = tokenizer = None

def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    db.execute('CREATE TABLE IF NOT EXISTS analyses (id TEXT PRIMARY KEY, text TEXT NOT NULL, sentiment TEXT NOT NULL, confidence REAL NOT NULL, probabilities TEXT NOT NULL, created_at TEXT NOT NULL, latency_ms REAL NOT NULL, token_count INTEGER NOT NULL, truncated INTEGER NOT NULL, source TEXT NOT NULL)')
    return db

def load_model():
    global model, tokenizer
    with lock:
        state.update(status='loading', detail='Loading local DistilBERT weights')
        try:
            paths = [MODEL_DIR] + sorted(MODEL_DIR.glob('checkpoint-*'), reverse=True)
            folder = next((p for p in paths if any((p / n).exists() for n in ['model.safetensors','pytorch_model.bin','model.safetensors.index.json','pytorch_model.bin.index.json'])), None)
            if folder is None:
                raise FileNotFoundError('Trained weights are missing. Restore model.safetensors or pytorch_model.bin to models/yelp-distilbert, then click Retry model. scheduler.pt and scaler.pt are training state, not model weights.')
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            token_folder = folder if (folder / 'tokenizer.json').exists() else ROOT / 'models/distilbert'
            tokenizer = AutoTokenizer.from_pretrained(token_folder, local_files_only=True)
            candidate = AutoModelForSequenceClassification.from_pretrained(folder, local_files_only=True)
            mapping = {'negatif':'negative','netral':'neutral','positif':'positive','negative':'negative','neutral':'neutral','positive':'positive'}
            # The supplied FinchW model card explicitly documents LABEL_0/1/2 as negative/neutral/positive.
            if folder.resolve() == (ROOT / 'models/yelp-distilbert').resolve():
                mapping.update({'label_0':'negative','label_1':'neutral','label_2':'positive'})
            actual = [mapping.get(str(candidate.config.id2label.get(i, '')).lower()) for i in range(3)]
            if candidate.config.num_labels != 3 or actual != LABELS:
                raise ValueError('Model label mapping must be 0=negative, 1=neutral, 2=positive.')
            state['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = candidate.to(state['device']).eval()
            state.update(status='ready', detail='Fine-tuned DistilBERT is ready')
        except Exception as exc:
            model = None
            state.update(status='unavailable', detail=str(exc))

@asynccontextmanager
async def lifespan(app):
    with connect():
        pass
    threading.Thread(target=load_model, daemon=True).start()
    yield

app = FastAPI(title='Tonewise Sentiment API', version='1.0.0', lifespan=lifespan)

class Review(BaseModel):
    text: str = Field(min_length=1, max_length=16000)
    source: str = Field(default='Manual', max_length=80)
    @field_validator('text')
    @classmethod
    def nonempty(cls, value):
        if not value.strip():
            raise ValueError('Enter a non-empty review')
        return value.strip()

class Batch(BaseModel):
    reviews: list[Review] = Field(min_length=1, max_length=200)

def predict(reviews):
    with lock:
        if state['status'] != 'ready' or model is None:
            raise HTTPException(503, state['detail'])
        import torch
        start = time.perf_counter()
        results = []
        for offset in range(0, len(reviews), 8):
            group = reviews[offset:offset+8]
            counts = [len(tokenizer.encode(r.text, truncation=False)) for r in group]
            tokens = tokenizer([r.text for r in group], return_tensors='pt', padding=True, truncation=True, max_length=512).to(state['device'])
            with torch.inference_mode():
                probabilities = model(**tokens).logits.softmax(-1).cpu().tolist()
            for review, probs, count in zip(group, probabilities, counts):
                index = max(range(3), key=lambda i: probs[i])
                results.append(dict(id=str(uuid.uuid4()), text=review.text, sentiment=LABELS[index], confidence=probs[index], probabilities=dict(zip(LABELS, probs)), created_at=datetime.now(timezone.utc).isoformat(), latency_ms=0, token_count=count, truncated=count>512, source=review.source))
        elapsed = round((time.perf_counter()-start)*1000/len(reviews), 2)
        with connect() as db:
            for row in results:
                row['latency_ms'] = elapsed
                db.execute('INSERT INTO analyses VALUES (?,?,?,?,?,?,?,?,?,?)', (row['id'],row['text'],row['sentiment'],row['confidence'],json.dumps(row['probabilities']),row['created_at'],elapsed,row['token_count'],int(row['truncated']),row['source']))
        return results

@app.get('/api/health')
def health():
    return {**state, 'model':'DistilBERT', 'max_tokens':512, 'repository':'FinchW/my-yelp-sentiment-model-finetuned'}

@app.post('/api/model/reload')
def reload_model():
    if state['status'] == 'loading':
        return health()
    state.update(status='loading', detail='Checking model files again')
    threading.Thread(target=load_model, daemon=True).start()
    return health()

@app.post('/api/analyze')
def analyze(review: Review):
    return predict([review])[0]

@app.post('/api/analyze/batch')
def batch(payload: Batch):
    return {'results': predict(payload.reviews)}

@app.get('/api/history')
def history():
    with connect() as db:
        rows = [dict(r) for r in db.execute('SELECT * FROM analyses ORDER BY created_at DESC')]
    for row in rows:
        row['probabilities'] = json.loads(row['probabilities'])
        row['truncated'] = bool(row['truncated'])
    return {'results': rows}

@app.get('/api/model/metrics')
def metrics():
    return json.loads((ROOT/'outputs/metrics/distilbert_metrics.json').read_text())

@app.get('/api/export')
def export():
    buffer = io.StringIO()
    fields = ['id','text','sentiment','confidence','created_at','source','latency_ms','token_count','truncated']
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    for row in history()['results']:
        row = {k: ("'"+v if isinstance(v,str) and v.lstrip().startswith(('=','+','-','@')) else v) for k,v in row.items()}
        writer.writerow(row)
    return Response(buffer.getvalue(), media_type='text/csv', headers={'Content-Disposition':'attachment; filename=tonewise-history.csv'})

if (ROOT/'dist').exists():
    app.mount('/', StaticFiles(directory=ROOT/'dist', html=True), name='dashboard')
