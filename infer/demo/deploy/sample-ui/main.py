from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from openai import OpenAI
import os

app = FastAPI()
client = OpenAI(
    base_url=f'http://{os.getenv("LITELLM_BASE_URL")}:4000',
    api_key=os.getenv('LITELLM_MASTER_KEY'),
)
conversation = []

HTML = """<!DOCTYPE html>
<html><head><title>Chat</title></head><body>
<div id="chat"></div>
<input id="input" type="text" style="width:80%"><button onclick="send()">Send</button>
<script>
async function send() {
    const msg = input.value; input.value = '';
    chat.innerHTML += '<p><b>You:</b> ' + msg + '</p>';
    chat.innerHTML += '<p><b>Bot:</b> ...</p>';
    const res = await fetch('/chat', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({message:msg})});
    const data = await res.json();
    chat.lastChild.innerHTML = '<b>Bot:</b> ' + data.response;
}
</script></body></html>"""


class ChatRequest(BaseModel):
    message: str


@app.get('/', response_class=HTMLResponse)
def index():
    return HTML


@app.post('/chat')
def chat(req: ChatRequest):
    conversation.append({'role': 'user', 'content': req.message})
    response = client.chat.completions.create(model='genollama', messages=conversation)
    reply = response.choices[0].message.content
    conversation.append({'role': 'assistant', 'content': reply})
    return {'response': reply}
