import json
import requests

payload={
    "jsonrpc":"2.0",
    "method":"add",
    "params":[10,33],
    "id":1
}
response=requests.post("http://localhost:4000",json=payload)

print("Response:",response.text)
print("Result:",response.json()["result"])