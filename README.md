# AI Games – Connect 4 Arena

**CSCI 487 Final Project**

In this project, I explored what it’s like to train an AI model to play Connect 4. Given only two days and zero prior AI experience, I dove headfirst into concepts like **backpropagation**, **reward structures**, and **neural network training** using **PyTorch**—all running on my RTX 3070 at home.

### 🧠 Neural Network Agent

The AI I trained takes in the **current game state** and outputs the **column index** it wants to play next. It wasn't particularly successful (time constraints + beginner mistakes), but it gave me a solid foundation for understanding how AI learns, especially in game environments.

### 🏁 Class Competition

The class was designed as a competition where we could pit our AIs against each other. It made the learning process super engaging and fun.

### 🤖 LLM Opponents (Bonus)

Dr. Caley also encouraged us to explore **LLM API calls**, so I added support for:

- **Claude**
- **ChatGPT**
- **DeepSeek**

Each of these large language models can also be challenged through the app. Performance-wise, they’re hit or miss—**DeepSeek** seemed to play surprisingly well, but it can take up to **6 minutes** to return a move.

### 🧱 Tech Stack

- **Frontend:** Vite + React
- **Backend:** Flask (Python)
- **ML:** PyTorch
