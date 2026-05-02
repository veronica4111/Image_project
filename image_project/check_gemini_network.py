import os
from dotenv import load_dotenv
import socket
import urllib.request
import urllib.error

load_dotenv()
print('PYTHON', __import__('sys').executable)
print('GEMINI_API_KEY configured:', bool(os.getenv('GEMINI_API_KEY')))
print('GEMINI_API_KEY prefix:', os.getenv('GEMINI_API_KEY')[:10] if os.getenv('GEMINI_API_KEY') else None)
for host in ['api.google.com', 'gemini.googleapis.com', 'ai.google.dev']:
    try:
        addrs = socket.getaddrinfo(host, 443)
        print(host, 'resolved', len(addrs), 'addresses first', addrs[0][4])
    except Exception as e:
        print(host, 'resolution error', type(e).__name__, e)

for url in ['https://api.google.com', 'https://gemini.googleapis.com', 'https://ai.google.dev']:
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as r:
            print(url, 'status', r.status)
    except urllib.error.HTTPError as e:
        print(url, 'HTTPError', e.code, e.reason)
    except Exception as e:
        print(url, 'error', type(e).__name__, e)
