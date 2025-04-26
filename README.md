# Enhanced ELIZA Chatbot

A modern Python implementation of the classic ELIZA psychotherapist chatbot, enhanced with state-of-the-art AI capabilities through Model Context Protocol (MCP), LangChain, and Ollama integration.

## Description

This project combines the classic ELIZA's Rogerian psychotherapist approach with modern AI technologies. It uses:
- Model Context Protocol (MCP) for standardized AI model interaction
- LangChain for advanced language model integration
- Ollama (with Gemma model) for enhanced natural language processing
- FastAPI for modern web API capabilities

## Features

- Natural language processing using pattern matching
- Modern responses for contemporary issues
- Support for various conversation topics including:
  - Anxiety and stress
  - Depression and mental health
  - Work-life balance
  - Technology and social media
  - Sleep issues
  - Loneliness
- Multiple greeting variations
- Empathetic and supportive responses

## Installation

1. Clone this repository:
```bash
git clone https://github.com/kasinadhsarma/eliza
cd eliza
```

2. Make sure you have Python 3.x installed.

3. Install Ollama and the Gemma model:
```bash
# Install Ollama (if not already installed)
curl https://ollama.ai/install.sh | sh

# Pull the Gemma model
ollama pull gemma3
```

4. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Usage

You can use ELIZA in two ways:

1. As a traditional command-line chatbot:
```bash
python3 eliza.py
```

2. As a modern API server with MCP support:
```bash
python3 enhanced_eliza.py
```

The enhanced version will start a server on port 8000 with the following endpoints:
- REST API: `http://localhost:8000/chat`
- MCP endpoint: Available through MCP client connections

Example API usage:
```python
import requests

response = requests.post("http://localhost:8000/chat", 
                        json={"user_input": "Hello, I'm feeling anxious"})
print(response.json()["response"])
```

Example MCP usage:
```python
from mcp.client import Client

async with Client() as client:
    response = await client.send("eliza.chat", input="Hello, I'm feeling anxious")
    print(response["response"])
```

To end a conversation, type any of these: "quit", "goodbye", or "bye".

## File Structure

- `eliza.py` - Main program file containing the ELIZA implementation
- `docter.txt` - Script file containing patterns and responses
- `README.md` - This file
- `LICENSE` - MIT License file

## How It Works

ELIZA works by:
1. Processing user input through pattern matching
2. Using decomposition rules to break down sentences
3. Applying reassembly rules to create responses
4. Using synonyms and keyword matching to understand context
5. Maintaining conversation flow with follow-up questions

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original ELIZA by Joseph Weizenbaum (1966)
- Inspired by the classic ELIZA implementations
- Enhanced with modern mental health awareness features
