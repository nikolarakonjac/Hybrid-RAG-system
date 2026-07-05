import requests

response = requests.post(
    "http://localhost:11434/api/generate",
    json={"model": "phi3", "prompt": "What is the capital of France?", "stream": False}
)

print(response.json()['response'])