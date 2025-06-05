from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
classifier = pipeline("text-classification", model=model_name)

app = FastAPI(title="Hugging Face Text Classification API")

class TextInput(BaseModel):
    text: str

@app.post("/classify")
async def classify_text(input: TextInput):
    result = classifier(input.text)
    
    return {
        "Текст": input.text,
        "Лейбл": result[0]["label"],
        "Точность": result[0]["score"]
    }


@app.get("/")
async def home():
    return {"message": "Hugging Face Text Classification API"}


# python -m uvicorn main:app --reload --port 8000
# curl -X POST "http://127.0.0.1:8000/classify" -H "Content-Type: application/json" -d "{\"text\":\"Hello\"}"