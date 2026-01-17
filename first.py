from fastapi import FastAPI
from pydantic import BaseModel

appn= FastAPI()

@appn.get("/")
def read_root():
    return {"first":"one","second":"two"}
    # return "Hello world!"


@appn.get("/abcd")
def start_call():
    return {"call_id":"CALL_123","status":"starting..."}

@appn.post("/status")
def status():
    return {"status":"running"}

@appn.get("/items/{item_id}")
def read_item(item_id):
    return {"Item id":item_id}

class Item(BaseModel):
    name: str
    email:str
@appn.post("/create-items/")
def create_item(item:Item):
    return "Name:"+item.name+" Email:"+item.email