👑 Quantum Queen AI

Quantum Queen AI is a hybrid AI project that combines conversational AI with quantum computing.

🚀 Features

- 👑 AI conversation powered by Qwen through Hugging Face Inference
- ⚛️ Quantum computing powered by Qiskit
- 🚦 Smart router for normal and quantum-related questions
- 🌐 FastAPI backend
- 💬 Conversation history support
- 🔐 Hugging Face token stored securely through environment variables
- ☁️ Designed for cloud deployment

🧠 How It Works

User
 ↓
FastAPI Backend
 ↓
Router
 ├── Normal Question → Queen AI
 │                      ↓
 │                  Qwen Model
 │
 └── Quantum Question → Qiskit
                         ↓
                    Quantum Result
                         ↓
                    Queen AI
                         ↓
                    Final Answer

📁 Project Structure

quantum-queen-ai/
├── main.py
├── router.py
├── queen.py
├── quantum.py
├── requirements.txt
└── README.md

⚙️ Technologies

- Python
- FastAPI
- Qwen
- Hugging Face Inference
- Qiskit
- Qiskit Aer
- Uvicorn

🔑 Environment Variables

Set the following variable on your hosting platform:

HF_TOKEN=your_huggingface_token

Do not put your Hugging Face token directly inside the source code.

▶️ Run

Install dependencies:

pip install -r requirements.txt

Start the server:

uvicorn main:app --host 0.0.0.0 --port $PORT

📌 Project Goal

The goal of Quantum Queen AI is to combine natural-language AI with quantum computing so that Queen AI can communicate with users while a quantum engine performs appropriate quantum computations.

⚠️ Note

The current quantum engine uses Qiskit simulation. A quantum-related question does not automatically mean that a quantum computer has solved the problem; the actual result depends on the quantum algorithm and circuit being executed.

---

Quantum Queen AI — AI + Quantum Computing ⚛️👑
