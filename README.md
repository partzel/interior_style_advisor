# 🪑 Interior Style Advisor – FastAPI Server

This is the backend server for the **Interior Style Advisor** project. It powers a .NET MAUI cross-platform application by providing interior design suggestions based on room images. It leverages a fine-tuned CLIP model and a language model to interpret room aesthetics and generate meaningful decor ideas.

---

## 🌐 Overview

This server accepts image uploads (e.g. living rooms, bedrooms) and returns interior decor suggestions. It uses a multimodal architecture combining:

- **CLIP** for visual understanding
- A **projector** to align vision and language spaces
- A **causal LLM** for generating design suggestions

The training data was scraped from interior design-related subreddits, with code and samples available in the [`notebooks/`](./notebooks/) directory.

---

## 🧠 Architecture

```text
User Image ➝ CLIP ➝ Projector ➝ LLM ➝ Decor Suggestion Text
````

Key components:

* `CLIP`: Extracts image features
* `Projector`: Transforms CLIP features into the LLM token space
* `LLM (Qwen2.5)` : Generates suggestions in natural language
* FastAPI: Exposes a `/caption` endpoint for the client to interact with

---

## 📦 Project Structure

```
.
├── app.py / routes.py      # FastAPI application
├── model/                  # Core model logic (CLIP, Projector, LLM)
├── data/                   # Dataset loading logic
├── scripts/                # Helper scripts (model loaders, etc.)
├── notebooks/              # Data scraping, training and experiments
├── requirements.txt        # Python dependencies
├── README.md               # You are here
```

---

## 🚀 Getting Started

### 1. Install dependencies

Create a virtual environment and install required packages:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the server

```bash
uvicorn app:app --reload
```

The server will be available at [http://localhost:8000](http://localhost:8000)

---

## 🛠️ API Usage

### `POST /caption`

**Description**: Accepts an image and an optional prompt, and returns a tailored interior decor suggestion.

**Request**:

* `image` (file): JPEG or PNG image of a room
* `prompt` (string, optional): Optional textual context (e.g. “for a cozy winter vibe”)

**Response**: Plain text with a suggestion

**Example using `curl`:**

```bash
curl -X POST http://localhost:8000/caption \
  -F "image=@./example_room.jpg" \
  -F "prompt=A relaxing bedroom look"
```

---

## 📚 Training & Dataset

Training data was collected from Reddit communities such as:

* `r/malelivingspace`
* `r/femalelivingspace`
* `r/DesignMyRoom`

Exploration and fine-tuning code lives in the [`notebooks/`](./notebooks/) folder, including:

* `living-space-datascraping.ipynb`
* `clip-model-training.ipynb`
* `llm-caption-roomdataset.ipynb`

---

## 🧾 License

MIT License. See [LICENSE](./LICENSE) for more details.

---

## 🙋‍♀️ Maintainers

This project was developed as part of the Deep Learning course followed at Higher School of Computer Science 08 May 1945 Sidi Bel Abbes broader.

For questions, please reach out or create a GitHub issue.