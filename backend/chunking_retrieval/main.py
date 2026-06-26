from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent import RANAssistantAgent

app = FastAPI(title="Telecom RAG API")

# Enable CORS so your React frontend running on localhost:5173 can access the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production/hackathon security, replace with actual origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the agent once when the server starts
print("Initializing Telecom Agent...")
agent = RANAssistantAgent()

class TicketRequest(BaseModel):
    issue: str

class TicketResponse(BaseModel):
    content: str
    logs: str
    specs: str

@app.post("/api/chat", response_model=TicketResponse)
async def process_ticket(request: TicketRequest):
    try:
        user_issue = request.issue
        
        # Step 1: Search and interpret logs
        raw_logs = agent.tool_search_logs(user_issue)
        log_context = agent.interpret_logs(raw_logs)
        
        # Step 2: Extract suspected failing protocol
        suspect_protocol = agent.extract_protocol_from_logs(log_context, user_issue)
        
        # Step 3: Gather specs
        spec_context = agent.tool_search_specs(suspect_protocol)
        
        # Step 4: Run reasoning loop prompt
        final_prompt = (
            f"You are a senior Telecom RAN engineer writing a fault report.\n\n"
            f"Reported issue: {user_issue}\n\n"
            f"Interpreted telemetry logs:\n{log_context}\n\n"
            f"3GPP specification context:\n{spec_context}\n\n"
            f"Suspected failing protocol: {suspect_protocol}\n\n"
            f"Write a short diagnostic report. Maximum 3 sentences per section.\n\n"
            f"### Executive Summary\n"
            f"### Root Cause Analysis\n"
            f"### Recommended Actions\n"
        )
        final_report = agent.call_llm(final_prompt, max_new_tokens=300)
        
        return {
            "content": final_report,
            "logs": log_context if log_context else "No critical anomalies found.",
            "specs": spec_context if spec_context else "No 3GPP specifications retrieved."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)