# ELIZA Chatbot

A Python implementation of the classic ELIZA psychotherapist chatbot, originally created by Joseph Weizenbaum in 1966.

## Description

ELIZA simulates a Rogerian psychotherapist by using pattern matching and substitution techniques to formulate responses to user inputs. This implementation includes modern additions and improvements to make the interactions more relevant to contemporary issues and mental health awareness.

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

## Usage

Run the program using Python:

```bash
python3 eliza.py
```

To end a conversation with ELIZA, type any of these: "quit", "goodbye", or "bye".

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
