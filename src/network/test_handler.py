import requests

data = {"key1": 0, "key2": 100}
response = requests.post('http://localhost:8000', data=data)
print(response.content)