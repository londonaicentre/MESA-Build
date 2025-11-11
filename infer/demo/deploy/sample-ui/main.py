import os, secrets
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from openai import OpenAI

from llm_assets.prompts import generate_system_prompt

load_dotenv()
app = FastAPI()
security = HTTPBasic()
client = OpenAI(
    base_url=f'http://{os.getenv("LITELLM_BASE_URL")}:4000',
    api_key=os.getenv('LITELLM_MASTER_KEY'),
)


class ChatRequest(BaseModel):
    message: str


def verify(credentials: HTTPBasicCredentials = Depends(security)):
    correct_user = secrets.compare_digest(
        credentials.username, os.getenv("AUTH_USER", "admin"))
    correct_pass = secrets.compare_digest(
        credentials.password, os.getenv("AUTH_PASS", "admin"))
    if not (correct_user and correct_pass):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, headers={
                            "WWW-Authenticate": "Basic"})
    return credentials.username


@app.get("/")
def index(user: str = Depends(security)):
    return FileResponse('index.html')


@app.post('/chat')
def chat(req: ChatRequest, user: str = Depends(verify)):
    conversation = [{
        "role": "system",
        "content": generate_system_prompt("systemprompt_finetune.md") + "\nThe document follows below:"
    }]
    conversation.append({'role': 'user', 'content': req.message})
    response = client.chat.completions.create(
        model='genollama', messages=conversation, temperature=0.1, max_tokens=8000)
    reply = response.choices[0].message.content
    conversation.append({'role': 'assistant', 'content': reply})
    return {'response': reply}
