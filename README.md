# 🕵️ Job Listings Extractor

> An AI-powered multi-agent system that takes a list of company names and autonomously hunts down their latest mid-to-senior level job postings — across job boards, career pages, and LinkedIn — then exports everything to a clean CSV.

---

## 🔴 The Problem

Business development, sales, and recruiting teams often need to monitor what roles competitor or target companies are actively hiring for. This tells you:

- Which companies are scaling (and where)
- What tech stacks or skill sets are in demand
- Where warm outreach opportunities exist

Doing this manually — Googling each company, checking their careers page, searching LinkedIn — is tedious and doesn't scale. A list of 50 companies means 50 separate searches, done by hand, every time.

---

## ✅ The Solution

**Job Listings Extractor** automates this entirely. You upload a CSV of company names, and a crew of AI agents gets to work — searching the web in real time, extracting job titles and locations, and handing back a structured, downloadable CSV.

No manual searching. No copy-pasting. Just upload and download.

---

## 🏗️ Architecture

```
User Uploads CSV (company names)
           │
           ▼
┌──────────────────────────────────┐
│         Streamlit Frontend        │
│  - CSV Upload                     │
│  - Spinner / Progress State       │
│  - Results Table + CSV Download   │
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│     Per-Company Agent Loop        │
│  For each company in CSV:         │
│                                   │
│  ┌────────────────────────────┐   │
│  │  Agent: Job Expert          │   │
│  │  Role: BDM (25 yrs exp)    │   │
│  │  Tool: SerperDevTool        │   │
│  │  (real-time Google search) │   │
│  └────────────┬───────────────┘   │
│               │                   │
│  ┌────────────▼───────────────┐   │
│  │  Task: Find 1 job posting   │   │
│  │  Output: Title, Location,   │   │
│  │  Company                    │   │
│  └────────────────────────────┘   │
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│     CrewAI Orchestration          │
│  - Crew.kickoff() runs all tasks  │
│  - Planning enabled (LLM plans    │
│    task order before execution)   │
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│     Output Parser (Regex)         │
│  Extracts structured fields from  │
│  raw agent output:                │
│  - Job Title                      │
│  - Job Location                   │
│  - Company                        │
└──────────────┬───────────────────┘
               │
               ▼
        Pandas DataFrame
        → Displayed in UI
        → Downloadable CSV
```

---

## 🤔 Why I Chose What I Chose

### CrewAI — for multi-agent orchestration
CrewAI makes it straightforward to define agents with specific roles, goals, and backstories, then assign them tasks and let them run. The "25-year BDM" backstory on the agent isn't just flavor — it shapes how the LLM reasons about what constitutes a relevant, senior-level job posting vs noise.

I chose CrewAI over a single prompt loop because:
- The `planning=True` flag lets the LLM reason about task ordering before executing, reducing hallucinated or irrelevant outputs
- Agents are reusable and composable — easy to add a second agent (e.g. a "Job Analyst" to summarize responsibilities) later
- Task outputs are structured and iterable via `crew.kickoff().tasks_output`

### SerperDevTool — for real-time web search
Job listings go stale fast. A static dataset or cached index won't cut it. Serper wraps Google Search with an API, giving the agent access to live results — the same results a human would find if they Googled the company name + "jobs" today.

`WebsiteSearchTool` and `CSVSearchTool` are imported but reserved for future enhancements (e.g. scraping a company's careers page directly, or cross-referencing the input CSV for additional metadata).

### GPT-4 / OpenAI — as the agent brain
CrewAI's default LLM backend is OpenAI. The agent needs strong instruction-following to reliably output `Job Title: X`, `Job Location: Y`, `Company: Z` in a parseable format. GPT-4 class models handle this consistently.

### Regex output parsing — intentionally simple
Agent outputs are natural language. Rather than asking the model to return JSON (which can break with longer outputs), I used simple regex patterns to extract the three fields. It's brittle against format drift, but for a constrained task with a tightly specified `expected_output`, it works reliably.

### Streamlit — for the UI
Fastest path to a usable, shareable interface. File upload, spinner, dataframe display, and a download button — all in ~20 lines. For a production tool this would be a proper web app, but for a portfolio/demo this is the right call.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- OpenAI API key — [platform.openai.com](https://platform.openai.com/api-keys)
- Serper API key — [serper.dev](https://serper.dev)

### Installation

```bash
git clone https://github.com/Touseeq99/job-listings-extractor.git
cd job-listings-extractor
pip install -r requirements.txt
```

### Environment Setup

Set your API keys as environment variables before running:

```bash
export OPENAI_API_KEY=your_openai_key_here
export SERPER_API_KEY=your_serper_key_here
```

Or create a `.env` file and load it with `python-dotenv`:

```env
OPENAI_API_KEY=your_openai_key_here
SERPER_API_KEY=your_serper_key_here
```

> ⚠️ Never hardcode API keys in `app.py`. Use environment variables only.

### Run

```bash
streamlit run app.py
```

---

## 📂 Input Format

Upload a CSV with a column named `Company Name`:

```csv
Company Name
Google
Stripe
Notion
Anthropic
```

---

## 📤 Output Format

A downloadable CSV with:

```csv
Job Title,Job Location,Company
Senior Software Engineer,London UK,Google
Product Manager,Remote,Stripe
...
```

---

## 📁 Project Structure

```
job-listings-extractor/
├── app.py              # Streamlit UI + CrewAI agent pipeline
├── requirements.txt    # Python dependencies
├── .gitignore
└── README.md
```

---

## ⚠️ Notes

- Processing time scales linearly with number of companies — each triggers a live web search + LLM call. For large lists (50+ companies), expect several minutes.
- Serper has a free tier with a limited number of searches per month. Check your usage at [serper.dev/dashboard](https://serper.dev/dashboard).
- Agent outputs are non-deterministic — results may vary slightly between runs.

---

## 🛠️ Built With

- [CrewAI](https://crewai.com/) — Multi-agent orchestration
- [Serper](https://serper.dev/) — Real-time Google Search API
- [OpenAI GPT-4](https://openai.com/) — Agent reasoning backbone
- [Streamlit](https://streamlit.io/) — Web UI
- [Pandas](https://pandas.pydata.org/) — Data handling

---

## 👤 Author

**Touseeq Ahmed**  
AI Engineer | [GitHub](https://github.com/Touseeq99)
