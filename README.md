# 🧬Artificial Intelligence in Medicine: Advances and Challenges in Blood Test Interpretation in Brazil

> **Your intelligent ally for interpreting CBCs, designed with real-world healthcare in mind.**

Welcome to our open-source web app that helps simplify the interpretation of Complete Blood Count (CBC) tests using Artificial Intelligence. Built with accessibility and Brazil's regional healthcare disparities in mind, this tool blends modern AI with medical knowledge to support both professionals and patients.

The project is developed using Streamlit and powered by cutting-edge Large Language Models (LLMs) and Natural Language Processing (NLP). It's currently being prepared for submission to **SBrT 2025 (XLIII Brazilian Symposium on Telecommunications and Signal Processing)**.

---

## 📚 Table of Contents

- [🚀 Project Overview](#-project-overview)
- [✨ Features](#-features)
- [🛠️ Technologies Used](#-technologies-used)
- [📦 Installation](#-installation)
- [▶️ How to Use](#-how-to-use)
- [📁 Data and Models](#-data-and-models)
- [🤝 Contributing](#-contributing)
- [📬 Contact](#-contact)

---

## 🚀 Project Overview

In many healthcare environments — especially underserved areas — interpreting blood tests like CBCs can be a complex task. What if a smart system could assist, offering clarity, reliability, and clinical insight?

That’s exactly what this app was built to do. It takes CBC values as input, analyzes them using AI, detects anomalies, and provides human-readable explanations backed by clinical knowledge. It's also tailored for the Brazilian medical context, including support for Portuguese documents and culturally relevant reference ranges.

Our mission? To assist medical professionals, streamline diagnostics, and foster healthcare equity through open-source AI.

---

## ✨ Features

What makes our app both powerful and easy to use:

- **🔵 Clean and Intuitive Interface**  
  Built with Streamlit — no complex setup needed. Open your browser, input the data, and you’re good to go.

- **📊 Smart Anomaly Detection**  
  Each result is assessed using gender-specific reference ranges and classified as low, normal, or high — with severity levels like mild, moderate, or severe.

- **🧠 AI-Generated Clinical Explanations**  
  Powered by models like BioMistral-7B, the app fetches relevant content from medical literature and translates it into clear, readable insights. When the AI can't provide an answer, fallback responses are used.

- **📝 Downloadable Reports**  
  Get a full report with summaries, highlighted findings, and recommendations — ready to download and share.

- **🧹 Modular and Scalable**  
  Expand it easily by adding new lab tests or medical documents.

---

## 🛠️ Technologies Used

We’ve chosen a modern tech stack to power the app:

- **Streamlit** – For the web UI
- **PyTorch** – To embed medical text
- **FAISS** – Fast document chunk retrieval
- **LangChain** – Connects LLMs to document workflows
- **HuggingFace Transformers** – For BioBERTpt-all (Portuguese biomedical model)
- **LlamaCpp** – Runs BioMistral-7B locally
- **PyPDF** – Extracts text from medical PDFs
- **NumPy** – Supports embeddings and similarity logic
- **Google Colab** – Main development environment with GPU

---

## 📦 Installation

To set it up locally:

1. **Clone the repository**
```bash
git clone https://github.com/AnnaBittencourt19/Artificial-Intelligence-in-Medicine-Advances-and-Challenges-in-Blood-Test-Interpretation.git
```

2. **Navigate into the folder**
```bash
cd Artificial-Intelligence-in-Medicine-Advances-and-Challenges-in-Blood-Test-Interpretation
```

3. **Install the dependencies**
```bash
pip install -r requirements.txt
```

---

## ▶️ How to Use

Once installed, you can launch the app using ngrok:

1. **(Optional) Kill existing ngrok tunnels**
```python
from pyngrok import ngrok
ngrok.kill()
```

2. **Start the app and expose it**
```python
from pyngrok import ngrok
import time

ngrok.set_auth_token("<your-ngrok-token>")  # Replace with your token
!streamlit run app.py &>/dev/null &
time.sleep(2)
public_url = ngrok.connect(8501)
print(f"🔗 Public app URL: {public_url}")
```

3. **Access the app**  
   Open the public ngrok URL or go to `http://localhost:8501`

4. **Enter the CBC test values**  
   Provide parameters like Hemoglobin, RBC, WBC, and patient gender

5. **Review the generated report**  
   The app will show a detailed output with medical explanations and suggestions

---

## 📁 Data and Models

You’ll need the following files:

- **Medical PDFs** – Place them in the `data/pdfs/` folder. They feed the knowledge base. [Data link](https://drive.google.com/drive/folders/1p6WvNubq7KfL-C4z5arQVMp7tiHY-oot?usp=sharing)
- **LLM File** – Download the `BioMistral-7B.Q4_K_M.gguf` model from [Hugging Face](https://huggingface.co/BioMistral/BioMistral-7B)

> **Note:** These were originally accessed via Google Drive, so local setup requires downloading and updating file paths accordingly.

---

## 🤝 Contributing

We welcome contributions! Here’s how to get involved:

1. Fork the repo
2. Create a branch for your feature or fix
3. Make your changes and document them well
4. Open a Pull Request
5. Wait for feedback from the maintainers

By contributing, you agree to the project’s license and code of conduct.

---

## 📬 Contact

Got a question, suggestion, or bug report? Reach out:

- 📧 Email: [annabittencourt279@gmail.com](mailto:annabittencourt279@gmail.com)
- 🐛 Open an issue on GitHub

Let’s build better healthcare tools — together. 💙

---

## ⚠️ Final Notes & Warnings

- 🔑 **Use your own HuggingFace token** for models/APIs that require it.
- 🌐 **Set up your ngrok token** if exposing the app online.
- 📁 **Ensure models and PDF files are in the correct folders** before launching.
- 🛠️ **Update file paths as needed**, depending on your environment.
- 💡 **This is not a diagnostic tool** — it’s designed to assist, not replace, licensed healthcare professionals.
