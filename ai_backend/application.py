from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.graph import neuraGraph

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
    response = neuraGraph.invoke(input={"messages": [("human", userInput)]})
    return response['final_response']

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)