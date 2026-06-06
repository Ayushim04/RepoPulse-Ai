# 🚀 RepoPulse AI - Your Open-Source Co-Maintainer

RepoPulse AI is an intelligent codebase assistant and automated Pull Request reviewer designed to supercharge open-source collaboration. Powered by Google's Gemini 1.5 Pro and a custom Retrieval-Augmented Generation (RAG) pipeline, it acts as a tireless co-maintainer—helping developers understand complex codebases and reviewing PRs in real-time.

Built for the **Open Source Hackathon 2026**.

---

## ✨ Key Features

* **🤖 Live PR Reviews (Webhook Integration):** Listens to GitHub PR events in real-time. The AI analyzes code diffs, catches potential bugs, and automatically posts constructive review comments directly on GitHub.
* **📚 Doc-Bot (RAG Knowledge Base):** A built-in chat assistant that ingests your repository's `.py`, `.js`, and `.md` files. Ask questions like *"Explain the webhook auth flow"* or *"How does the RAG pipeline work?"* and get accurate, context-aware answers.
* **⚡ Beautiful Developer Dashboard:** A sleek, dark-themed UI built with Tailwind CSS that displays live PR review feeds, RAG chunking status, and system KPIs.
* **🔒 Secure by Design:** Validates GitHub webhook payloads using `X-Hub-Signature-256` to ensure only authentic requests trigger the AI engine.

---

## 🛠️ Tech Stack

* **Backend:** Python, Flask
* **AI Engine:** Google Gemini 1.5 Pro (Generative AI API)
* **Architecture:** Retrieval-Augmented Generation (RAG), Vector Chunking
* **Frontend:** HTML5, Vanilla JS, Tailwind CSS
* **Integration:** GitHub Webhooks, PyGithub (for API interactions)

---

## ⚙️ Local Setup & Installation

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/repopulse-ai.git](https://github.com/your-username/repopulse-ai.git)
cd repopulse-ai
```

### 2. Install Dependencies
Ensure you have Python installed, then install the required packages:
```bash
pip install -r requirements.txt
pip install python-dotenv
```

### 3. Environment Variables
Create a .env file in the root directory based on the provided .env.example file.
```bash
Code snippet
FLASK_ENV=development
FLASK_SECRET_KEY=repopulse-dev-secret-do-not-use-in-prod
GITHUB_TOKEN=ghp_your_personal_access_token_here
GITHUB_WEBHOOK_SECRET=hackathon2026secret
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-1.5-pro-latest
REPO_PATH=.
LOG_LEVEL=DEBUG
```

### 4. Run the Flask Server
```bash
python app.py
The dashboard will be live at http://127.0.0.1:5000. 
```

## 🔗 Connecting to GitHub (Live Webhooks)
To test the live PR review feature locally, we expose the local Flask server to the internet using **ngrok**.

### 1) Start the Flask server
```bash
python app.py
```

### 2) Start ngrok (port 5000)
In a new terminal:

```bash
ngrok http 5000
```
Copy the generated public URL (example):
`https://moonscape-litmus-taste.ngrok-free.dev`

### 3) Configure GitHub Webhook
Go to **Your GitHub Repository → Settings → Webhooks → Add webhook**.

Set:
- **Payload URL:** `https://your-ngrok-url.ngrok-free.dev/api/webhook/github`
- **Content type:** `application/json`
- **Secret:** `hackathon2026secret` *(Must match `GITHUB_WEBHOOK_SECRET` in your `.env`)*
- **Events:** Select **“Let me select individual events”** and check only **Pull requests**

Click **Add webhook** (or **Update webhook**).

## 🧠 Generating the RAG Index
Before the Doc-Bot can answer questions, it must ingest your repository code.

1. Open the dashboard in your browser
2. In the left sidebar, under **SYSTEM**, click **Sync Knowledge Base**
3. Wait for the success popup (the system will read files and generate vector chunks)
4. Start chatting with the codebase

## 🚀 What's Next?
- **Cloud Deployment:** Move from local ngrok tunneling to a fully hosted AWS/GCP environment.
- **Multi-Repo Support:** Let organizations manage multiple repositories from one dashboard.
- **Auto-Fix PRs:** Beyond reviewing—allow Gemini to commit suggested fixes back to the PR branch.

Made with ❤️ for the Open Source Community.
