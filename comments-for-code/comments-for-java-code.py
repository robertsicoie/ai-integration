# imports

import json
import os
import tempfile
from dotenv import load_dotenv
from openai import OpenAI
import anthropic
import gradio as gr
import requests


# Load environment variables from .env import keys for OpenAI and Anthropic API from .env file
# Needed for OpenAI and Anthropic API models only
load_dotenv(override=True)
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')
os.environ['ANTHROPIC_API_KEY'] = os.getenv('ANTHROPIC_API_KEY', 'your-key-if-not-using-env')

openai = OpenAI()
claude = anthropic.Anthropic()
OPENAI_MODEL = "gpt-4o"
CLAUDE_MODEL = "claude-3-5-sonnet-20240620"
LOCAL_DEEPSEEK_CODER = "deepseek-coder:latest"
LOCAL_DEEPSEEK_LLAMA_3_2 = "llama3.2:1b"

OLLAMA_API = "http://localhost:11434/api/chat"
# To start Ollama server, run:
# ollama serve
#
# To install local models, you can use:
# ollama pull deepseek-coder:latest
# ollama pull llama3.2:1b


# low cost models
# OPENAI_MODEL = "gpt-4o-mini"
# CLAUDE_MODEL = "claude-3-haiku-20240307"

system_message = "You are an assistant that adds commments and Javadoc to Java code. "
system_message += "Add only relevant comments, explaining what methods and code blocks do. "
system_message += "If there already exist comments in the code, replace them if you have better suggestions. "
system_message += "Give me the Java class only, without any additional text before or after the class."

def user_prompt_for(code):
    user_prompt = "Add Javadoc and comments to the Java code bellow. "
    user_prompt += code
    return user_prompt

def messages_for(code):
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt_for(code)}
    ]

def stream_comments_for(code, model):
    print(f"Streaming comments for code using model: {model}")

    if model == "GPT":
        result = gpt_stream_comments_for(code)
    elif model == "Claude":
        result = claude_stream_comments_for(code)
    elif model == LOCAL_DEEPSEEK_LLAMA_3_2 or model == LOCAL_DEEPSEEK_CODER:
        result = local_stream_comments_for(code, model)
    else:
        raise ValueError(f"Unknown model: {model}") 

    for chunk in result:  
        yield chunk 

def gpt_stream_comments_for(code):
    stream = openai.chat.completions.create(model=OPENAI_MODEL, messages=messages_for(code), stream=True)
    reply = ""
    for chunk in stream:
        fragment = chunk.choices[0].delta.content or ""
        reply += fragment
        yield reply.replace('{"model":"llama3.2:1b","created_at":"2025-07-31T11:46:48.784082Z","message":{"role":"assistant","content":"','').replace('```','')
    
def local_stream_comments_for(code, model):
    messages = {
        "model": model,
        "messages": messages_for(code),
        "stream": True
    }

    response = requests.post(OLLAMA_API, json=messages, headers={"Content-Type": "application/json"}, stream=True)
    reply = ""
    for line in response.iter_lines():
        if line:
            fragment = line.decode('utf-8')
            fragment = json.loads(fragment).get("message", {}).get("content", "")
            print(fragment)
            reply += fragment
        
            yield reply

def claude_stream_comments_for(code):
    result = claude.messages.stream(
        model = CLAUDE_MODEL,
        max_tokens = 2000,
        system = system_message,
        messages = [{"role": "user", "content": user_prompt_for(code)}]
    )

    reply = ""
    with result as stream:
        for text in stream.text_stream:
            reply += text
            yield reply.replace('```java\n','').replace('```','')

def prepare_download(content):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".java", mode='w', encoding='utf-8') as temp_file:
        temp_file.write(content)
        return temp_file.name
    
def load_uploaded_file(uploaded_file):
    print(f"Loading file: {uploaded_file}")
    print(f"File size: {os.path.getsize(uploaded_file)} bytes")
    with open(uploaded_file.name, 'r') as file:
        yield file.read()

with gr.Blocks() as ui:
    
    gr.Markdown("## Comments for Java code")
    gr.Markdown("This app adds comments to Java code using LLMs. "
                "You can select between OpenAI and local Deepseek models.")
    with gr.Row():
        file_input = gr.File(label="Upload Java file", file_types=[".java"])
        download_file = gr.File(label="Download commented code")
    with gr.Row():
        initial = gr.Textbox(label="Java code:", lines=30, placeholder="Paste or upload your Java code here")
        commented = gr.Textbox(label="Java code with comments:", lines=10)
    with gr.Row():
        model = gr.Dropdown([OPENAI_MODEL, CLAUDE_MODEL, LOCAL_DEEPSEEK_LLAMA_3_2, LOCAL_DEEPSEEK_CODER], label="Select model", value=LOCAL_DEEPSEEK_LLAMA_3_2)
        convert_btn = gr.Button("Comment code")
        download_btn = gr.Button("Download")

    file_input.change(load_uploaded_file, inputs=[file_input], outputs=[initial])
    convert_btn.click(stream_comments_for, inputs=[initial, model], outputs=[commented])
    download_btn.click(prepare_download, inputs=[commented], outputs=[download_file])

ui.launch(inbrowser=True)