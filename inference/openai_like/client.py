import requests

url = 'http://localhost:6006/chat'

headers = {
    'Host': 'localhost:6006',
    'User-Agent': 'python-requests/2.24.0',
    'Accept': '*/*',
    'Content-Type': 'application/json',
}

data = {
    "stream": True,
    "chunk": False,
    "top_p": 1,
    "temperature": 0.7,
    "max_tokens": 4096,
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
        {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
        {"role": "user","content": "Where was it played?"}
    ]
}

response = requests.post(url, headers=headers, json=data)

print(response.text)
