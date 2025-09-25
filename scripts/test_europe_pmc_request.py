import requests
import json

def probe(query='sleep'):
    url = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search'
    params = {'query': query, 'format': 'json', 'pageSize': 1, 'page': 1}
    print('Request URL:', url)
    print('Params:', params)
    r = requests.get(url, params=params, timeout=30)
    print('Status:', r.status_code)
    try:
        j = r.json()
        print('Keys:', list(j.keys()))
        print('hitCount:', j.get('hitCount'))
        rl = j.get('resultList', {}).get('result', [])
        print('resultList len:', len(rl))
        if rl:
            print('sample id/title:', rl[0].get('id'), rl[0].get('title'))
    except Exception as e:
        print('Failed to parse JSON:', e)
        print('Text sample:', r.text[:1000])

if __name__ == '__main__':
    probe('sleep')
