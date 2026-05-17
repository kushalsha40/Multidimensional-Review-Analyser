from fastapi import FastAPI
from pydantic import BaseModel

from src.pipeline import analyze_review

app = FastAPI()

# input schema
class ReviewRequest(BaseModel):
    review:str

#Health check route
@app.get("/")
def home():
    return {"message": "Welcome to the IMDB Review Analysis API!"}

@app.post("/analyze")
def analyze(request:ReviewRequest):
    result = analyze_review(request.review)
    return result
