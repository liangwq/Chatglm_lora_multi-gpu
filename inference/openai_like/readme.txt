curl --location --request POST 'http://localhost:6006/chat' \
--header 'Host: localhost:6006' \
--header 'User-Agent: python-requests/2.24.0' \
--header 'Accept: */*' \
--header 'Content-Type: application/json' \
--data-raw '{  "stream": true,"chunk": false,"top_p": 1,"temperature": 0.7,"max_tokens": 4096,  "messages": [{"role": "system","content": "You are a helpful assistant."},{"role": "user","content": "Who won the world series in 2020?"}]}'

