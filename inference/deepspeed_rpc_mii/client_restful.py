import json
import requests
url = f"http://localhost:28080/mii/mistral-deployment"
params = {"prompts": ["DeepSpeed is", "Seattle is"], "max_length": 128}
json_params = json.dumps(params)
output = requests.post(
    url, data=json_params, headers={"Content-Type": "application/json"}
)
