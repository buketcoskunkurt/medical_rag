import requests, json
url='http://127.0.0.1:8000/retrieve'
body={'question':'Uyku sorunları neden olur?','k':5}
res = requests.post(url,json=body,timeout=20)
print('status', res.status_code)
print(json.dumps(res.json(), ensure_ascii=False, indent=2))
