import requests
import json

BASE_URL = 'http://127.0.0.1:5000'

data = {'question':'Name'}

# POST request
response = requests.post("{}/predict".format(BASE_URL), data = data)

# print results
print(response)
print(response.status_code)
print(response.json())
