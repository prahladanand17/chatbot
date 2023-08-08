from fastapi import (
    FastAPI, WebSocket, WebSocketDisconnect,
    Request, UploadFile, File
)
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

from fastapi.templating import Jinja2Templates
import json
from socket_1 import SocketManager
import PyPDF2
from langchain.tools import BaseTool, StructuredTool
from typing import Any, List

from langchain import OpenAI
import langchain
langchain.debug=True
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.schema.retriever import BaseRetriever, Document
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.callbacks.streaming_stdout_final_only import (
    FinalStreamingStdOutCallbackHandler,
)



from typing import Any, List
from langchain.schema import Document
from langchain.callbacks.manager import Callbacks


manager = SocketManager()
app = FastAPI()
templates = Jinja2Templates(directory="templates")
#llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613", verbose=True, streaming=True, callbacks=[StreamingStdOutCallbackHandler()])
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613", verbose=True)

input_documents = []

class DocumentSearchInput(BaseModel):
    query: str = Field()

class DocumentSearchTool(BaseTool):
    def run(query: str) -> str:
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="map_reduce", retriever=CustomRetriever())
        #import pdb; pdb.set_trace()
        return qa.run(query)
        
    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")
    
class CustomRetriever(BaseRetriever):
    def get_relevant_documents(
        self, query: str, *, callbacks: Callbacks = None, **kwargs: Any
    ) -> List[Document]:
        sample_documents = []
        sample_documents.append(input_documents[0])
        #sample_documents.append(input_documents[8])
        
        return input_documents
        
    async def aget_relevant_documents(
        self, query: str, *, callbacks: Callbacks = None, **kwargs: Any
    ) -> List[Document]:
        """Asynchronously get documents relevant to a query.
        Args:
            query: string to find relevant documents for
            callbacks: Callback manager or list of callbacks
        Returns:
            List of relevant documents
        """
        raise NotImplementedError();

@app.get("/")
def hello(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.get('/chatwindow')
def open_chat(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@app.post("/upload")
async def upload_file(file: UploadFile):
    return parse_file(file)
    
def parse_file(file):
    try:
        file_object = file.file
        pdfReader = PyPDF2.PdfReader(file_object)
        
        numPages = len(pdfReader.pages)
        for i in range(numPages):
            page = pdfReader.pages[i]
            page_text = page.extract_text()
                        
            global input_documents
            input_documents.append(Document(page_content=page_text, metadata={'page': i + 1}))

    except:
        return json.dumps("Error uploading file")
        
    return json.dumps("Upload Success!")

@app.websocket("/chat")
async def chat(websocket: WebSocket):
    #import pdb; pdb.set_trace()

    tools = [ StructuredTool.from_function(
                func=DocumentSearchTool.run,
                name = "DocumentSearchQuestionAnswering",
                description = "Use this tool when you are given a query and to answer this query you need to search over a document or documents that are not readily available on the internet. Do not use this tool for general conversation",
                args_schema=DocumentSearchInput
            )]
    
    agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)
    
    await manager.connect(websocket)
    response = {
        "message": "got connected"
    }
    await manager.broadcast(response) 
    
    try:
        while True:
            query = await websocket.receive_json()
            await manager.broadcast(query)
            response = agent({"input": query})
            await manager.broadcast(
                agent({"input": query})
            )
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        response['message'] = "left"
        await manager.broadcast(response)