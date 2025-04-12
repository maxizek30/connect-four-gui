import { useState } from "react";
import "./App.css";

const ROWS = 6;
const COLS = 7;

function App() {
  const backgroundAudio = new Audio("chill.mp3");

  const [board, setBoard] = useState(
    Array(ROWS)
      .fill(0)
      .map(() => Array(COLS).fill(0))
  );
  const [currentPlayer, setCurrentPlayer] = useState(1); // 1 = Human, -1 = AI
  const [winner, setWinner] = useState(null);
  const [aiType, setAiType] = useState("local"); // "local", "deepseek", "claude", or "chatgpt"
  const [error, setError] = useState(null);

  const humanSound = new Audio("quak.mp3");
  const aiSound = new Audio("bark.mp3");
  const humanWinSound = new Audio("cry.mp3");
  const aiWinSound = new Audio("yay.mp3");

  const aiImages = {
    local:
      "https://a.espncdn.com/combiner/i?img=/i/headshots/nba/players/full/1966.png",
    deepseek: "Subject.png",
    claude:
      "https://gallerypngs.com/wp-content/uploads/2024/12/Chill-Guy-Png-Photo-Free-Download.png", // Replace with your Claude AI image URL
    chatgpt: "https://i.redd.it/l9cby426oap71.png",
  };

  // Determine valid columns where a piece can be dropped
  const getValidColumns = (board) => {
    return board[0]
      .map((cell, colIndex) => (cell === 0 ? colIndex : null))
      .filter((col) => col !== null);
  };

  // Handle human player's click
  const handleClick = async (col) => {
    if (winner || currentPlayer !== 1) return;

    setError(null); // Clear any previous error
    const newBoard = board.map((row) => [...row]);

    // Place the human player's piece
    for (let row = ROWS - 1; row >= 0; row--) {
      if (newBoard[row][col] === 0) {
        newBoard[row][col] = currentPlayer;
        humanSound.play();
        break;
      }
    }

    // Log the new board after human move
    console.log("Human Move - Updated Board:", newBoard);

    // Check if human just won
    if (checkWin(newBoard, currentPlayer)) {
      setBoard(newBoard);
      setWinner("Human");
      humanWinSound.play(); // Play human win sound
      return;
    }

    // Switch to AI's turn
    setBoard(newBoard);
    setCurrentPlayer(-1);

    // Let the AI make a move
    try {
      const aiMove = await getAIMove(newBoard);
      for (let row = ROWS - 1; row >= 0; row--) {
        if (newBoard[row][aiMove] === 0) {
          newBoard[row][aiMove] = -1; // AI's piece
          aiSound.play();
          break;
        }
      }

      // Check if AI just won
      if (checkWin(newBoard, -1)) {
        setBoard(newBoard);
        setWinner("AI");
        aiWinSound.play(); // Play AI win sound
        return;
      }

      // Switch back to human
      setBoard(newBoard);
      setCurrentPlayer(1);
    } catch (err) {
      setError("Failed to fetch AI move. Please try again.");
    }
  };

  // Decide which backend endpoint to call based on the selected AI type
  const getAIMove = async (currentBoard) => {
    const validCols = getValidColumns(currentBoard);

    let endpoint;
    switch (aiType) {
      case "local":
        endpoint = "http://127.0.0.1:5000/ai_move";
        break;
      case "deepseek":
        endpoint = "http://127.0.0.1:5000/deepseek_move";
        break;
      case "claude":
        endpoint = "http://127.0.0.1:5000/claude_move";
        break;
      case "chatgpt":
        endpoint = "http://127.0.0.1:5000/chatgpt_move";
        break;
      default:
        endpoint = "http://127.0.0.1:5000/ai_move";
    }

    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ board: currentBoard, valid_cols: validCols }),
    });

    if (!response.ok) {
      throw new Error(`AI request failed with status: ${response.status}`);
    }

    const data = await response.json();
    if (data.error) {
      throw new Error(data.error);
    }

    return data.move;
  };

  // Let the AI move first
  const handleBotFirst = async () => {
    resetGame();
    const newBoard = [...board];
    try {
      const aiMove = await getAIMove(newBoard);
      for (let row = ROWS - 1; row >= 0; row--) {
        if (newBoard[row][aiMove] === 0) {
          newBoard[row][aiMove] = -1; // AI's piece
          aiSound.play();
          break;
        }
      }
      setBoard(newBoard);
      setCurrentPlayer(1);
    } catch (err) {
      setError("Failed to fetch AI move. Please try again.");
    }
  };

  // Reset the game state
  const resetGame = () => {
    setBoard(
      Array(ROWS)
        .fill(0)
        .map(() => Array(COLS).fill(0))
    );
    setCurrentPlayer(1);
    setWinner(null);
    setError(null);
  };

  // Check for a winning condition
  const checkWin = (board, player) => {
    return (
      checkHorizontal(board, player) ||
      checkVertical(board, player) ||
      checkDiagonal(board, player)
    );
  };

  // Check horizontally
  const checkHorizontal = (board, player) => {
    for (let row = 0; row < ROWS; row++) {
      for (let col = 0; col <= COLS - 4; col++) {
        if (
          board[row][col] === player &&
          board[row][col + 1] === player &&
          board[row][col + 2] === player &&
          board[row][col + 3] === player
        ) {
          return true;
        }
      }
    }
    return false;
  };

  // Check vertically
  const checkVertical = (board, player) => {
    for (let col = 0; col < COLS; col++) {
      for (let row = 0; row <= ROWS - 4; row++) {
        if (
          board[row][col] === player &&
          board[row + 1][col] === player &&
          board[row + 2][col] === player &&
          board[row + 3][col] === player
        ) {
          return true;
        }
      }
    }
    return false;
  };

  // Check diagonally (both directions)
  const checkDiagonal = (board, player) => {
    // Top-left to bottom-right
    for (let row = 0; row <= ROWS - 4; row++) {
      for (let col = 0; col <= COLS - 4; col++) {
        if (
          board[row][col] === player &&
          board[row + 1][col + 1] === player &&
          board[row + 2][col + 2] === player &&
          board[row + 3][col + 3] === player
        ) {
          return true;
        }
      }
    }
    // Bottom-left to top-right
    for (let row = 3; row < ROWS; row++) {
      for (let col = 0; col <= COLS - 4; col++) {
        if (
          board[row][col] === player &&
          board[row - 1][col + 1] === player &&
          board[row - 2][col + 2] === player &&
          board[row - 3][col + 3] === player
        ) {
          return true;
        }
      }
    }
    return false;
  };

  return (
    <div className="connect4-container">
      <div className="top-left-controls">
        <button onClick={() => backgroundAudio.play()} className="play-button">
          Play Background Music
        </button>
      </div>
      <h1>Connect 4</h1>
      {winner ? (
        <h2>{winner} Wins!</h2>
      ) : (
        <p>Current Player: {currentPlayer === 1 ? "Human" : "AI"}</p>
      )}

      {error && <p className="error">{error}</p>}

      <div className="boardContainer">
        <div className="board">
          {board.map((row, rowIndex) => (
            <div key={rowIndex} className="row">
              {row.map((cell, colIndex) => (
                <div
                  key={colIndex}
                  className="cell"
                  onClick={() => handleClick(colIndex)}
                  style={{
                    backgroundColor:
                      cell === 1 ? "red" : cell === -1 ? "yellow" : "white",
                  }}
                ></div>
              ))}
            </div>
          ))}
        </div>
      </div>

      <button onClick={resetGame} className="reset-button">
        Reset Game
      </button>
      <button onClick={handleBotFirst} className="reset-button">
        Bot Goes First
      </button>

      {/* Radio buttons for AI selection (only one can be selected) */}
      <div className="ai-toggle">
        <label>
          <input
            type="radio"
            name="aiType"
            value="local"
            checked={aiType === "local"}
            onChange={() => setAiType("local")}
          />
          Local AI
        </label>
        <label>
          <input
            type="radio"
            name="aiType"
            value="deepseek"
            checked={aiType === "deepseek"}
            onChange={() => setAiType("deepseek")}
          />
          DeepSeek AI
        </label>
        <label>
          <input
            type="radio"
            name="aiType"
            value="claude"
            checked={aiType === "claude"}
            onChange={() => setAiType("claude")}
          />
          Claude AI
        </label>
        <label>
          <input
            type="radio"
            name="aiType"
            value="chatgpt"
            checked={aiType === "chatgpt"}
            onChange={() => setAiType("chatgpt")}
          />
          ChatGPT AI
        </label>
      </div>

      {/* Conditionally display the AI image based on the selected aiType */}
      <div className="ai-image-container">
        <img src={aiImages[aiType]} alt={aiType} className="ai-image" />
      </div>
    </div>
  );
}

export default App;
