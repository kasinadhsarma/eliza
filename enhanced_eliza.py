"""
Enhanced ELIZA using Model Context Protocol (MCP) and LangChain with Ollama integration
"""
import os
from typing import Dict, List, Optional, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, ConfigDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

from mcp.server.fastmcp import FastMCP


class ConversationContext(BaseModel):
    """Conversation context for enhanced ELIZA"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    history: List[Dict[str, str]] = []
    user_profile: Dict[str, Any] = {}
    current_topic: Optional[str] = None
    sentiment: Optional[str] = None


class EnhancedEliza:
    def __init__(self):
        # Initialize Ollama with the Gemma model
        self.llm = OllamaLLM(model="gemma3")
        self.context = ConversationContext()

        # Initialize chat template
        self.template = ChatPromptTemplate.from_template('''
You are ELIZA, a sophisticated AI therapist that combines classic ELIZA's Rogerian
approach with modern therapeutic techniques. Consider the following conversation
history and respond empathetically:

Conversation History:
{history}

Current User Input: {user_input}

Respond as ELIZA:''')

        # Create the chain
        self.chain = self.template | self.llm

    async def process_input(self, user_input: str) -> str:
        """Process user input using advanced NLP and LLM capabilities"""
        # Add user input to conversation history
        self.context.history.append({"role": "user", "content": user_input})

        # Build a string of the last 5 messages
        history_str = "\n".join([
            f"{'User' if msg['role'] == 'user' else 'ELIZA'}: {msg['content']}"
            for msg in self.context.history[-5:]
        ])

        # Generate response via LangChain
        response = await self.chain.ainvoke({
            "history": history_str,
            "user_input": user_input
        })

        processed_response = str(response)
        self.context.history.append({"role": "assistant", "content": processed_response})

        return processed_response


# --- Application setup ---
app = FastAPI()

# Initialize MCP server (named "eliza")
mcp = FastMCP("eliza")

# Instantiate ELIZA logic
eliza = EnhancedEliza()

# HTTP endpoint for simple REST clients
@app.post("/chat")
async def chat_http(user_input: str):
    """HTTP POST /chat → {"response": ...}"""
    return {"response": await eliza.process_input(user_input)}


# MCP tool registration: this will expose `eliza_chat` as a callable tool over /mcp
@mcp.tool()
async def eliza_chat(input: str) -> Dict[str, str]:
    """
    eliza.chat tool → takes `input` (the user’s message) and returns {"response": ...}.
    """
    response = await eliza.process_input(input)
    return {"response": response}


if __name__ == "__main__":
    # Enable CORS for the HTTP API
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount the MCP server under /mcp
    app.mount("/mcp", mcp)

    # Run the combined FastAPI + MCP app
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
