import json
import requests

method=input("Enter method (add/subtract): ")
payload={
    "jsonrpc":"2.0",
    "method":method,
    "params":[5,3],
    "id":1
}
if(method=="add"):
    response=requests.post("http://localhost:4000",json=payload)
else:
    response=requests.post("http://localhost:5000",json=payload)

print("Response:",response.text)
print("Result:",response.json()["result"])