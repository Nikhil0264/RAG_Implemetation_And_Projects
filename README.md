# College Assistant & LangGraph Workflows

A practical collection of AI workflow examples built with **LangGraph** and **LangChain**. The main application is a Streamlit-based College Assistant that answers student questions about academic rules and fees using the supplied PDF documents. The repository also includes focused examples of sequential pipelines, parallel branches, conditional retrieval, tool use, and human-in-the-loop review.

## Highlights

- Document-grounded answers for academic and fee-related questions
- Separate retrieval paths for the academics handbook and fee structure
- Programme-aware responses for BCA, BBA, and B.Com (H) students
- Conversational Streamlit interface with source passages available for inspection
- Reusable LangGraph patterns: sequential execution, parallel reducers, conditional routing, tool calls, and approval interrupts

## Primary application: College Assistant

The browser application in `app.py` is the recommended starting point. It loads two PDF documents, creates an in-memory FAISS vector index for each one, and uses a LangGraph workflow to answer a student’s question.

### Workflow

```text
Student question
       |
       v
Classifier (academic / fee / general)
       |
       +--- academic --> Search academics_handbook.pdf ---+
       |                                                  |
       +--- fee ------> Search IITM_Fees_Structure_SAMPLE.pdf --> Response generator --> Answer
       |                                                  |
       +--- general --> No document retrieval -----------+
```

1. The user selects a programme and submits a question.
2. `classifier_node` categorizes the question as `academic`, `fee`, or `general`.
3. `route_query` sends academic and fee questions to their corresponding retriever. General questions skip retrieval.
4. The appropriate retriever searches the four most relevant chunks from its PDF index.
5. `response_node` asks the Groq model to answer using the retrieved context and selected programme.
6. The interface displays the reply and, when document retrieval was used, the source passages in an expandable panel.

The PDF indexes are built once per Streamlit process through `@st.cache_resource`, keeping later questions responsive. They are held in memory; no external vector database is required.

## Getting started

### Prerequisites

- Python 3.10 or later
- A Groq API key
- Internet access on first run to download the embedding model

### Installation

```bash
git clone <your-repository-url>
cd Langgraph
python -m venv .venv
```

Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

```bash
pip install -r requirements.txt
pip install streamlit
```

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Never commit `.env`; it is intentionally excluded by `.gitignore`.

### Run the web app

```bash
streamlit run app.py
```

Open the local address printed by Streamlit, choose a programme in the sidebar, and ask a question such as:

> What is the minimum attendance required to sit for exams?

Both `academics_handbook.pdf` and `IITM_Fees_Structure_SAMPLE.pdf` must remain alongside `app.py`.

## Repository guide

| File | Purpose |
| --- | --- |
| `app.py` | Main Streamlit College Assistant with a polished chat interface and streamed workflow status. |
| `conditional_RAG.py` | Terminal version of the College Assistant workflow. |
| `sequential_base.py` | Three-stage editor → scriptwriter → Hinglish translator pipeline. |
| `parallel_reducers.py` | Parallel safety analysis branches merged with a custom state reducer. |
| `iterative_tools.py` | LinkedIn post writer that can search the web, self-review, and revise up to three times. |
| `humaninloop.py` | LinkedIn post writer that pauses for human approval or feedback with LangGraph interrupts. |
| `state.py` | Small state-schema examples using `TypedDict` and Pydantic validation. |
| `academics_handbook.pdf` | Source document for academic-policy retrieval. |
| `IITM_Fees_Structure_SAMPLE.pdf` | Source document for fee-related retrieval. |
| `requirements.txt` | Core Python dependencies. |

## Workflow examples in detail

### 1. Sequential content pipeline — `sequential_base.py`

This example demonstrates a linear state graph:

```text
START → editor → scriptwriter → translator → END
```

- `editor_node` corrects grammar, spelling, and tone while preserving the source message.
- `scriptwriter_node` turns the edited text into an engaging video-script hook.
- `translator_node` adapts that script to natural Hinglish.

Each node writes one new field into `pipelineState`, making the output of one stage the input to the next.

### 2. Parallel analysis with a reducer — `parallel_reducers.py`

Three nodes run independently from `START`:

- `toxicity_node` scores profanity, aggression, hate speech, and toxicity.
- `copyright_node` scores originality, plagiarism, and trademark risk.
- `culture_node` scores cultural or regional sensitivity risks.

Every branch returns a partial `safety_scores` dictionary. The `merge_score_dicts` reducer combines those updates instead of allowing one branch to overwrite another. The result contains all three scores after the graph reaches `END`.

### 3. Conditional RAG — `conditional_RAG.py`

This is the command-line predecessor of the Streamlit app. It demonstrates conditional edges: a classifier directs each query to an academic retriever, fee retriever, or general-answer route. It is useful for understanding the graph logic without the UI layer.

### 4. Tool-assisted iterative writing — `iterative_tools.py`

The LinkedIn writer can call `web_search` through a LangGraph `ToolNode` when a topic needs current information. A separate reviewer checks the generated post against defined quality rules. Rejected posts return to the writer with reviewer feedback; the workflow ends after approval or three writing attempts.

This example additionally requires these environment variables:

```env
HF_API_KEY=your_hugging_face_token
SEARCH_API_KEY=your_serpapi_key
GROQ_API_KEY=your_groq_api_key
```

Install its additional dependency if you run it:

```bash
pip install google-search-results
```

### 5. Human-in-the-loop approval — `humaninloop.py`

This workflow replaces automated review with a real reviewer. `human_review_node` calls LangGraph’s `interrupt`, pausing the graph after each draft. The terminal asks the reviewer to type `approved` or provide revision feedback; the workflow resumes with `Command(resume=...)`. `MemorySaver` preserves the graph state between the pause and resume operations.

This script requires:

```env
HF_API_KEY=your_hugging_face_token
```

## Key technical choices

- **LangGraph `StateGraph`:** Models each workflow as explicit nodes and edges, making control flow easy to inspect and extend.
- **Typed state:** Each workflow defines a `TypedDict` state shape, so data passed between nodes is clear and predictable.
- **FAISS + MiniLM embeddings:** PDFs are split into overlapping 800-character chunks and searched semantically with `sentence-transformers/all-MiniLM-L6-v2`.
- **Grounded generation:** Academic and fee answers are prompted with retrieved document passages rather than relying only on model knowledge.
- **Groq chat model:** The College Assistant uses `ChatGroq` with `openai/gpt-oss-120b` for classification and response generation.
- **State reducers:** The parallel example uses `Annotated` state to merge concurrent dictionary updates safely.
- **Checkpointed interruption:** The human-review example uses `MemorySaver` to make a paused graph resumable.

## Project structure

```text
Langgraph/
├── app.py
├── conditional_RAG.py
├── sequential_base.py
├── parallel_reducers.py
├── iterative_tools.py
├── humaninloop.py
├── state.py
├── academics_handbook.pdf
├── IITM_Fees_Structure_SAMPLE.pdf
├── requirements.txt
└── .gitignore
```

## Notes and limitations

- The answer quality depends on the accuracy and completeness of the two PDFs. Update those files when the underlying policies or fees change.
- FAISS indexes are rebuilt when the app process starts. For larger document collections, persist the index or use a managed vector store.
- The classifier is model-based, so ambiguous questions may be routed imperfectly. Add deterministic rules or confirmation prompts if that becomes important.
- The included API keys belong only in `.env`; use `.env.example` with placeholder values if you want to share configuration guidance publicly.

## License

No license has been specified yet. Add a license file before publishing or reusing the project publicly.
