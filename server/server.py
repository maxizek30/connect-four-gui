from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import numpy as np
import torch
import logging
from environment import Connect4Env
from model import QNetwork
from openai import OpenAI
from dotenv import load_dotenv
import os
import anthropic

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
CHATGPT_API_KEY = os.getenv("CHATGPT_API_KEY")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = QNetwork(num_actions=7).to(device)
model.load_state_dict(torch.load("models/connect4_model_282000.pth", map_location=device))
model.eval()

# DeepSeek API Configuration
# DeepSeek API Configuration
DEEPSEEK_API_URL = "https://api.deepseek.com"
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_API_URL)


# OpenAI API Configuration for ChatGPT
chatgpt_client = OpenAI(api_key=CHATGPT_API_KEY)

# Anthropic API Configuration
claude_client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

# Endpoint for local AI move
@app.route("/ai_move", methods=["POST"])
def ai_move():
    data = request.json
    board = np.array(data["board"])
    valid_cols = data["valid_cols"]

    # Neural network inference
    board_t = torch.from_numpy(board).float().unsqueeze(0).unsqueeze(0).to(device)
    q_values = model(board_t)[0].cpu().detach().numpy()
    best_valid_col = valid_cols[np.argmax([q_values[c] for c in valid_cols])]

    return jsonify({"move": best_valid_col})


@app.route("/deepseek_move", methods=["POST"])
def deepseek_move():
    logging.debug("Received request at /deepseek_move")
    try:
        # Get board state and valid columns from request
        data = request.json
        logging.debug(f"Request data: {data}")
        board = data["board"]
        valid_cols = data["valid_cols"]

        # Construct the prompt for DeepSeek Reasoner
        prompt = f"""
        You are playing Connect 4 as the AI. The board is a 6x7 grid. Each cell can be:
        - 1 (Human's piece)
        - -1 (AI's piece)
        - 0 (Empty space)
        
        The board is represented as a list of rows (top to bottom). Columns are numbered 0-6 from left to right.
        Only choose from the following valid columns: {valid_cols}.
        
        Return the number of the column where you would drop your piece. Do not add any other text.

        Board:
        {board}
        """
        logging.debug(f"Prompt sent to DeepSeek Reasoner: {prompt}")

        # Make the API request using OpenAI client
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[{"role": "user", "content": prompt}],
        )

        # Extract reasoning content and final move
        reasoning_content = response.choices[0].message.reasoning_content
        content = response.choices[0].message.content.strip()
        logging.debug(f"Reasoning Content: {reasoning_content}")
        logging.debug(f"Move Content: {content}")

        # Parse the move from the response
        move = int(content)
        if move in valid_cols:
            logging.info(f"DeepSeek Reasoner selected column: {move}")
            return jsonify({"move": move, "reasoning": reasoning_content})
        else:
            logging.warning(f"Invalid move returned by DeepSeek Reasoner: {move}")
            return jsonify({"error": "Invalid move returned by DeepSeek Reasoner"}), 400

    except Exception as e:
        logging.error(f"Error in /deepseek_move: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/claude_move", methods=["POST"])
def claude_move():
    logging.debug("Received request at /claude_move")
    try:
        # Get board state and valid columns from request
        data = request.json
        logging.debug(f"Request data: {data}")
        board = data["board"]
        valid_cols = data["valid_cols"]

        # Construct the user instructions
        user_instructions = f"""
        You are playing Connect 4 as an AI player. The board is a 6x7 grid represented as a list of rows (top to bottom), where:
        - 1 represents a Human's piece.
        - -1 represents the AI's piece.
        - 0 represents an empty space.

        Columns are numbered 0-6 from left to right. Only choose from the following valid columns: {valid_cols}.
        Return only the column number where you would drop your piece. Do not add any other text.

        Board:
        {board}
        """

        logging.debug(f"Prompt sent to Claude: {user_instructions}")

        # Call Anthropic's Messages API
        response = claude_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            messages=[{
                "role": "user",
                "content": user_instructions
            }],
            temperature=0,
            max_tokens=100
        )

        # Extract the response content
        content = response.content[0].text.strip()
        logging.debug(f"Claude's Response: {content}")

        # Parse the move
        move = int(content)
        if move in valid_cols:
            logging.info(f"Claude selected column: {move}")
            return jsonify({"move": move})
        else:
            logging.warning(f"Invalid move returned by Claude: {move}")
            return jsonify({"error": "Invalid move returned by Claude"}), 400

    except Exception as e:
        logging.error(f"Error in /claude_move: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500
    logging.debug("Received request at /claude_move")
    try:
        data = request.json
        logging.debug(f"Request data: {data}")
        board = data["board"]
        valid_cols = data["valid_cols"]

        # Construct the user instructions
        user_instructions = f"""
        You are playing Connect 4 as an AI player. The board is a 6x7 grid represented as a list of rows (top to bottom), where:
        - 1 represents a Human's piece.
        - -1 represents the AI's piece.
        - 0 represents an empty space.

        Columns are numbered 0-6 from left to right. Only choose from the following valid columns: {valid_cols}.
        Return only the column number where you would drop your piece. Do not add any other text.

        Board:
        {board}
        """

        logging.debug(f"Prompt sent to Claude: {user_instructions}")

        # Build the actual prompt for Anthropic
        # We combine anthropic.HUMAN_PROMPT, user_instructions, and anthropic.AI_PROMPT.
        prompt = f"{anthropic.HUMAN_PROMPT}{user_instructions}{anthropic.AI_PROMPT}"

        # Call Anthropic's completion API
        response = claude_client.completions.create(
        model="claude-3-5-sonnet-20241022",                # Or "claude-1", "claude-instant-1", etc.
            prompt=prompt,                   # Must be a single string
            max_tokens_to_sample=100,        # Required parameter
            temperature=0,                   # Temperature if desired
            stream=False                     # Whether to stream the response
        )

        # The completion text
        content = response.completion.strip()
        logging.debug(f"Claude's Response: {content}")

        # Parse the move
        move = int(content)
        if move in valid_cols:
            logging.info(f"Claude selected column: {move}")
            return jsonify({"move": move})
        else:
            logging.warning(f"Invalid move returned by Claude: {move}")
            return jsonify({"error": "Invalid move returned by Claude"}), 400

    except Exception as e:
        logging.error(f"Error in /claude_move: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500
    logging.debug("Received request at /claude_move")
    try:
        # Get board state and valid columns from request
        data = request.json
        logging.debug(f"Request data: {data}")
        board = data["board"]
        valid_cols = data["valid_cols"]

        # Construct the prompt for Claude
        prompt = f"""
    You are an advanced AI playing Connect 4. The game rules are as follows:
    - The board is a 6x7 grid. Rows are numbered top-to-bottom (0 to 5), and columns are numbered left-to-right (0 to 6).
    - Players alternate dropping pieces into columns. Pieces fall to the lowest available space in the selected column.
    - The goal is to get four consecutive pieces in a row (horizontally, vertically, or diagonally).

    Your role:
    - You are playing as the AI (-1). The opponent (Human) is represented by 1.
    - Empty spaces are represented as 0.

    Objectives:
    1. If you can win by placing a piece, prioritize that move.
    2. If the opponent can win in their next turn, block their move.
    3. If no immediate win or block is available, choose a move that maximizes your chances of winning in future turns.
    4. Play strategically both offensively and defensively.

    Constraints:
    - Only choose from the following valid columns: {valid_cols}.
    - Return ONLY the column number where you would drop your piece. Do not include any additional text or explanations.

    Current Board:
    {board}
    """
        logging.debug(f"Prompt sent to Claude: {prompt}")

        # Make the API request to Claude
        response = claude_client.completions.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=100,
            temperature=0,
            messages=[
                {"role": "user", "content": prompt},
            ],
        )
        content = response.completion.strip()
        logging.debug(f"Claude's Response: {content}")

        # Parse the move from Claude's response
        move = int(content)
        if move in valid_cols:
            logging.info(f"Claude selected column: {move}")
            return jsonify({"move": move})
        else:
            logging.warning(f"Invalid move returned by Claude: {move}")
            return jsonify({"error": "Invalid move returned by Claude"}), 400

    except Exception as e:
        logging.error(f"Error in /claude_move: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/chatgpt_move", methods=["POST"])
def chatgpt_move():
    logging.debug("Received request at /chatgpt_move")
    try:
        data = request.json
        board = data["board"]          # The current board state, 6x7
        valid_cols = data["valid_cols"]  # e.g. [0, 1, 2, 3, 4, 5, 6]

        # Create a very detailed prompt, just like your other endpoints
        # but include a strong instruction at the end to return ONLY the column number.
        user_prompt = f"""
    You are an advanced AI playing Connect 4. The game rules are as follows:
    - The board is a 6x7 grid. Rows are numbered top-to-bottom (0 to 5), and columns are numbered left-to-right (0 to 6).
    - Players alternate dropping pieces into columns. Pieces fall to the lowest available space in the selected column.
    - The goal is to get four consecutive pieces in a row (horizontally, vertically, or diagonally).

    Your role:
    - You are playing as the AI (-1). The opponent (Human) is represented by 1.
    - Empty spaces are represented as 0.

    Objectives:
    1. If you can win by placing a piece, prioritize that move.
    2. If the opponent can win in their next turn, block their move.
    3. If no immediate win or block is available, choose a move that maximizes your chances of winning in future turns.
    4. Play strategically both offensively and defensively.

    Constraints:
    - Only choose from the following valid columns: {valid_cols}.
    - Return ONLY the column number where you would drop your piece. Do not include any additional text or explanations.

    Current Board:
    {board}
    """

        # Call the GPT‑4O model. Adjust the model name to "gpt-4o" or "gpt-4o-mini" as allowed by your account.
        response = chatgpt_client.chat.completions.create(
            model="gpt-4o-mini",  # or "gpt-4o" if available
            messages=[
                # A system message helps enforce the "only respond with an integer" rule
                {
                    "role": "system",
                    "content": (
                        "You are a helpful AI that plays Connect 4. "
                        "When the user asks for a move, you MUST return only a single integer. "
                        "Do not include any other words, punctuation, or lines."
                    )
                },
                # The user message with the detailed prompt
                {"role": "user", "content": user_prompt},
            ],
            max_completion_tokens=500,
            temperature=0  # 0 for deterministic, or tweak if desired
        )

        # Extract the raw text from the response
        raw_content = response.choices[0].message.content.strip()
        logging.debug(f"GPT-4O raw response: {raw_content}")

        # In case the model still returns extra text, parse out the first integer you find
        import re
        match = re.search(r"\b\d+\b", raw_content)
        if not match:
            raise ValueError(
                f"Could not find a valid integer in GPT-4O response: {raw_content}"
            )
        move = int(match.group(0))  # Convert the found text to an integer

        # Verify that the integer is in our valid columns
        if move in valid_cols:
            logging.info(f"GPT-4O selected column: {move}")
            return jsonify({"move": move})
        else:
            logging.warning(f"GPT-4O returned an invalid column: {move}")
            return jsonify({"error": "Invalid move returned by GPT-4O"}), 400

    except Exception as e:
        logging.error(f"Error in /chatgpt_move: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500





if __name__ == "__main__":
    app.run(debug=True)
