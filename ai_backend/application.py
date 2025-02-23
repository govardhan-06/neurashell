from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.graph import neuraGraph
from langchain_core.messages import HumanMessage

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/v1/ask")
async def ask(userInput: str):
    # Create a proper message object
    message = HumanMessage(content=userInput)
    response = neuraGraph.invoke(input={"messages": [message]})
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)