# pip install fastapi uvicorn transformers sentencepiece jinja2

from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5Tokenizer, T5ForConditionalGeneration
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import re

# Initialize FastAPI app
app = FastAPI(title='Text Summarization System', description="Summarize dialogues", version="1.0")

# Load model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("./saved_summary_model")
tokenizer = T5Tokenizer.from_pretrained("./saved_summary_model")

# Ensure the model is on the correct device
device = "cuda" if model.device.type == "cuda" else "cpu" # "cuda" -> GPU
model = model.to(device)

# Mount Templates
templates = Jinja2Templates(directory="templates")

# Input Schema for requests
class DialogueInput(BaseModel):
    dialogue: str

def clean_text(text):
    text = re.sub(r'\r\n', ' ', text) # remove carriage returns and line breaks 
    text = re.sub(r'\s+', ' ', text) # remove extra spaces
    text = re.sub(r'<.*?>', '', text) # remove any xml/html tags
    text = text.strip().lower() # removes end spaces and covert to lower case
    return text
    # we do not remove stop words because we have to summarize the text

def summarize_dialogue(dialogue):
    dialogue = clean_text(dialogue)
    inputs = tokenizer(dialogue, return_tensors='pt', truncation=True, padding='max_length', max_length=512)

    inputs = {key: values.to(device) for key, values in inputs.items()}

    outputs = model.generate(
        inputs['input_ids'],
        max_length = 150,
        num_beams = 4,
        early_stopping = True
    )

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

# API Endpoint for text summarization
@app.post('/summarize/')
async def summarize(dialogue_input: DialogueInput):
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {'summary': summary}

# HTML UI
@app.get('/', response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})


