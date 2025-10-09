# server.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ---- Load model ----
MODEL_NAME = "EleutherAI/gpt-neo-125M"  # small GPT-Neo model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# ---- FastAPI app ----
app = FastAPI()

class Query(BaseModel):
    question: str

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/ask")
def ask_question(q: Query):
    input_ids = tokenizer.encode(q.question, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return {"answer": answer}
import uvicorn

if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
