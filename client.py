# client.py
import requests

API_URL = "http://127.0.0.1:8000/ask"

question = input("Ask something: ")
response = requests.post(API_URL, json={"question": question})
print("AI says:", response.json()["answer"])
