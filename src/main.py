"""
Generative Agent Onboarding API Gateway Entrypoint
===================================================

Main entry point of the FastAPI backend application. Handles connection
lifecycles, CORS middleware registration, and routes.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from config.logging_config import get_logger
from src.utils.api_client import ResilientLLMClient, LLMClientError
from src.core.memory import MemoryStreamManager
from src.pipelines.interview import compile_interview_graph
from src.api.router import router, llm_client_exception_handler, general_exception_handler

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Lifespan Context Manager
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles global server startup and shutdown hooks."""
    load_dotenv()
    logger.info("Initializing FastAPI Lifespan startup routines")

    # 1. Instantiate the global LLM Client
    llm_client = ResilientLLMClient()

    # 2. Extract and check Supabase credentials
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY")

    memory_manager = None
    if supabase_url and supabase_key:
        logger.info("Credentials found. Initializing MemoryStreamManager client singleton.")
        memory_manager = MemoryStreamManager(
            llm_client=llm_client,
            supabase_url=supabase_url,
            supabase_key=supabase_key,
        )
        try:
            await memory_manager.initialize()
            logger.info("MemoryStreamManager persistent client initialized successfully.")
        except Exception as e:
            logger.exception("Failed to initialize persistent Supabase client pool")
    else:
        logger.warning(
            "SUPABASE_URL or keys not configured. MemoryStreamManager created in fallback mode."
        )
        # Create uninitialized manager so that tests or manual overrides can proceed without crashing.
        try:
            memory_manager = MemoryStreamManager(
                llm_client=llm_client,
                supabase_url="https://placeholder.supabase.co",
                supabase_key="placeholder-key",
            )
        except Exception:
            pass

    # 3. Cache the compiled LangGraph workflow instance
    logger.info("Compiling LangGraph interview workflow.")
    graph = compile_interview_graph(llm_client, memory_manager)

    # Cache everything in app state
    app.state.llm_client = llm_client
    app.state.memory_manager = memory_manager
    app.state.graph = graph

    logger.info("FastAPI initialization completed successfully.")
    yield

    # 4. Teardown: Clean up connections
    logger.info("Starting FastAPI Lifespan shutdown routines.")
    if memory_manager:
        try:
            client = memory_manager._client
            if client and hasattr(client, "aclose"):
                await client.aclose()
                logger.info("Supabase client connection pool closed cleanly.")
        except Exception as e:
            logger.error("Error closing Supabase client pool during teardown: %s", e)

    logger.info("FastAPI shutdown completed.")


# ---------------------------------------------------------------------------
# App Instantiation & Middleware Configuration
# ---------------------------------------------------------------------------


app = FastAPI(
    title="Generative Agent Onboarding API",
    description=(
        "Asynchronous API Gateway wrapping our LangGraph State Machine "
        "and Supabase Three-Factor Memory Stream."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Policy: Only allow known frontend origins when credentials are enabled
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://life-decision-simulation-engine.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes router
app.include_router(router)

# Register Custom Exception Handler Boundaries
app.add_exception_handler(LLMClientError, llm_client_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)


@app.get("/")
def read_root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Generative Agent Onboarding API",
        "version": "1.0.0",
    }
