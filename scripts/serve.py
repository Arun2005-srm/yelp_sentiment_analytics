"""Start the local dashboard and open the browser once the server responds."""
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root), str(root/'.runtime')]
try:
    import uvicorn
except ImportError:
    print('Install dependencies first: python -m pip install -r requirements.txt')
    raise SystemExit(1)

def open_when_ready():
    for _ in range(90):
        try:
            with urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=1) as response:
                if response.status == 200:
                    webbrowser.open('http://127.0.0.1:8000')
                    return
        except OSError:
            time.sleep(1)

if __name__ == '__main__':
    threading.Thread(target=open_when_ready, daemon=True).start()
    uvicorn.run('backend.main:app', host='127.0.0.1', port=8000)
