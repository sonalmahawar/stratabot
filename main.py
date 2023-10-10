from fastapi import FastAPI, UploadFile, File, Form, Depends
from fastapi.responses import HTMLResponse
from typing import List
import shutil
import os
import config
import pandas as pd

 
# Initialize global variables

global_index = None
global_service_context = None
session_context_map = {}


# Set OpenAI API key environment variable

os.environ["OPENAI_API_KEY"] = config.API_DETAILS["openai_api_key"]

# Initialize FastAPI app

app = FastAPI()

 
# Initialize your LLM and other components here

from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index import ServiceContext
from llama_index.llms import OpenAI

llm = OpenAI(temperature=0, max_tokens=512, model="gpt-4")

def initialize_index_and_context():

    global global_index, global_service_context
    service_context = ServiceContext.from_defaults(llm=llm)
    documents = SimpleDirectoryReader('docs').load_data()
    index = VectorStoreIndex.from_documents(documents)
    global_index = index
    global_service_context = service_context

 

# Initialize index and context at the start

initialize_index_and_context()

 

def get_query_engine():
    global global_index, global_service_context
    return global_index.as_query_engine(service_context=global_service_context)

 
@app.get("/")

def read_root():
    with open('index.html', 'r') as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

 

@app.post("/uploadfile/")

async def create_upload_files(files: List[UploadFile] = File(...)):

    for file in files:
        file_location = f"docs/{file.filename}"
        with open(file_location, "wb+") as buffer:
            shutil.copyfileobj(file.file, buffer)

 
    # Update the global index after uploading new files

    initialize_index_and_context()
    return {"message": "File uploaded successfully"}

 

 

@app.delete("/deletefile/{filename}/")

async def delete_file(filename: str):

    try:
        os.remove(f"docs/{filename}")
        initialize_index_and_context()
        return {"message": "File deleted successfully"}

    except Exception as e:
        return {"message": str(e)}

 

 

@app.get("/fetchfiles/")
async def fetch_files():
    files = os.listdir('docs')
    return {"files": files}

 
@app.post("/text/")
async def process_text(text: str = Form(...), session_id: str = Form(...)):
    if session_id not in session_context_map:

         # Initialize new context
       session_context_map[session_id] = {}  # Or whatever context you want to initialize
    current_context = session_context_map[session_id]
    query_engine = get_query_engine()
    response = query_engine.query(text)

     # Save the updated context back into session_context_map

    session_context_map[session_id] = current_context

    return {"text": f": {response}"}