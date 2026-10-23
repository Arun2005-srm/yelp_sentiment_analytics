import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend import main as m

@pytest.fixture(scope='module')
def client(tmp_path_factory):
    m.DB_PATH = tmp_path_factory.mktemp('tonewise')/'test.sqlite3'
    with TestClient(m.app) as client:
        m.load_model()
        assert m.state['status']=='ready', m.state
        yield client

def test_validation(client):
    for text in ['', '   ', 'x'*16001]:
        assert client.post('/api/analyze',json={'text':text}).status_code==422
    assert client.post('/api/analyze/batch',json={'reviews':[]}).status_code==422
    assert client.post('/api/analyze/batch',json={'reviews':[{'text':'fine'}]*201}).status_code==422

def test_real_predictions_and_history(client):
    for text,label in [('The food was absolutely delicious and the service was wonderful!','positive'),('Disgusting food, rude staff. A terrible waste of money.','negative')]:
        response=client.post('/api/analyze',json={'text':text})
        assert response.status_code==200,response.text
        row=response.json()
        assert row['sentiment']==label
        assert sum(row['probabilities'].values())==pytest.approx(1,abs=1e-5)
        assert 0<=row['confidence']<=1
        assert row['latency_ms']>0
        assert any(r['id']==row['id'] for r in client.get('/api/history').json()['results'])

def test_batch_truncation_export(client):
    response=client.post('/api/analyze/batch',json={'reviews':[{'text':'Good service!','source':'CSV import'},{'text':'It was okay. '*700},{'text':'=SUM(1,2)'}]})
    assert response.status_code==200,response.text
    rows=response.json()['results']
    assert len(rows)==3
    assert rows[1]['truncated'] and rows[1]['token_count']>512
    assert rows[0]['source']=='CSV import'
    export=client.get('/api/export')
    assert export.status_code==200
    assert "'=SUM(1,2)" in export.text
    assert len(client.get('/api/history').json()['results'])==5

def test_missing_weights_does_not_fake_predictions(client,monkeypatch,tmp_path):
    previous_model=m.model
    previous_state=dict(m.state)
    monkeypatch.setattr(m,'MODEL_DIR',tmp_path)
    try:
        m.load_model()
        assert m.state['status']=='unavailable'
        assert client.post('/api/analyze',json={'text':'Great!'}).status_code==503
        assert len(client.get('/api/history').json()['results'])==5
    finally:
        m.model=previous_model
        m.state.update(previous_state)

def test_metrics_and_dashboard(client):
    assert client.get('/api/model/metrics').json()['test']['accuracy']==.9235
    assert client.get('/').status_code==200
    assert 'Tonewise' in client.get('/').text
