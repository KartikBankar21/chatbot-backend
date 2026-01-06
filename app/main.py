"""
FastAPI Application - Main Entry Point
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn

from app.config import config
from app.schemas import ChatMessage, ChatResponse, SessionInfo
from app.model_manager import model_manager
from app.inference import inference_pipeline
from app.session_manager import session_manager


# ============================================================================
# Startup/Shutdown
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle management"""
    # Startup
    print("="*80)
    print("🚀 Starting Chatbot Backend")
    print("="*80)
    
    # Load model
    try:
        model_manager.load_model(
            checkpoint_path=config.MODEL_CHECKPOINT,
            mappings_path=config.MAPPINGS_PATH
        )
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        raise
    
    yield
    
    # Shutdown
    print("\n🛑 Shutting down...")


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="Multi-turn NLU Chatbot API",
    description="Context-aware chatbot with dialogue state tracking",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "running",
        "model_loaded": model_manager.loaded,
        "supported_domains": config.SUPPORTED_DOMAINS
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage, background_tasks: BackgroundTasks):
    """
    Main chat endpoint
    
    Processes user message and returns:
    - Intent
    - Extracted slots
    - Requested slots
    - Response (for supported domains only)
    """
    try:
        # Get or create session
        session_id = message.session_id
        if not session_id:
            session_id = session_manager.create_session()
        
        # Process message
        result = inference_pipeline.process_message(
            message=message.message,
            session_id=session_id
        )
        
        # Schedule session cleanup in background
        background_tasks.add_task(session_manager.cleanup_expired_sessions)
        
        return ChatResponse(**result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.get("/session/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str):
    """Get session information"""
    info = session_manager.get_session_info(session_id)
    if not info:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    return SessionInfo(**info)


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session"""
    session_manager.delete_session(session_id)
    return {"status": "deleted", "session_id": session_id}


@app.post("/session/reset/{session_id}")
async def reset_session(session_id: str):
    """Reset a session (clear history but keep session active)"""
    session_manager.delete_session(session_id)
    new_session_id = session_manager.create_session()
    return {"status": "reset", "new_session_id": new_session_id}


@app.get("/domains")
async def get_domains():
    """Get supported domains and their schemas"""
    from app.schemas import DOMAIN_SCHEMAS
    return {
        "supported_domains": config.SUPPORTED_DOMAINS,
        "schemas": {k: v for k, v in DOMAIN_SCHEMAS.items() if k in config.SUPPORTED_DOMAINS}
    }


# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=config.HOST,
        port=config.PORT,
        reload=True  # Set to False in production
    )
