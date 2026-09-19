import os
import time
import streamlit as st
from typing import TypedDict,Annotated
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph,START,END
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(
    page_title="College Assistant",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Silences the "Examining the path of transformers.models... torchvision" log spam.
# Streamlit's file watcher inspects every imported module, and some lazy
# `transformers` modules need torchvision. The app itself never uses them, so
# the watcher is told to skip them. (If your Streamlit version differs, this
# is a harmless no-op - see the note about installing torchvision.)
# -----------------------------------------------------------------------------
try:
    from streamlit.watcher import local_sources_watcher as _lsw

    if not getattr(_lsw.get_module_paths, "_patched", False):
        _orig_get_module_paths = _lsw.get_module_paths

        def _get_module_paths(module):
            try:
                name = module.__name__
            except Exception:
                name = ""
            if name.startswith("transformers"):
                return set()
            return _orig_get_module_paths(module)

        _get_module_paths._patched = True
        _lsw.get_module_paths = _get_module_paths
except Exception:
    pass


# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,500;6..72,600&family=Public+Sans:wght@400;500;600&display=swap');

.stApp, .stApp p, .stApp li, .stApp label, .stApp textarea,
.stApp input, .stApp button, .stApp summary {
    font-family: 'Public Sans', system-ui, -apple-system, 'Segoe UI', sans-serif;
}
.block-container, [data-testid="stMainBlockContainer"] {
    max-width: 820px;
    padding-top: 2rem;
    padding-bottom: 6rem;
}
footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }

/* ---------- Masthead: a ruled notebook page with a margin line ---------- */
.masthead {
    position: relative;
    overflow: hidden;
    background-color: #0F1E3D;
    background-image: repeating-linear-gradient(
        to bottom,
        transparent 0, transparent 31px,
        rgba(255,255,255,0.055) 31px, rgba(255,255,255,0.055) 32px
    );
    border-radius: 6px;
    padding: 1.7rem 1.8rem 1.5rem 4.6rem;
    margin-bottom: 1.5rem;
}
.masthead::before {
    content: "";
    position: absolute;
    top: 0; bottom: 0; left: 2.9rem;
    width: 2px;
    background: rgba(229, 72, 77, 0.65);
}
.masthead-title {
    font-family: 'Newsreader', Georgia, serif;
    font-weight: 600;
    font-size: 2.5rem;
    line-height: 1.08;
    letter-spacing: -0.01em;
    color: #FFFFFF !important;
}
.masthead-sub {
    margin-top: 0.5rem;
    max-width: 32rem;
    font-size: 0.98rem;
    line-height: 1.5;
    color: #B7C3DB !important;
}
.masthead-meta { margin-top: 1rem; display: flex; flex-wrap: wrap; gap: 0.5rem; }
.pill {
    display: inline-flex; align-items: center; gap: 0.45rem;
    padding: 0.18rem 0.8rem;
    border-radius: 999px;
    font-size: 0.8rem; font-weight: 500;
    background: rgba(255,255,255,0.09);
    color: #E3EAF8 !important;
}
.pill-prog { background: #E5484D; color: #FFFFFF !important; font-weight: 600; }
.dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
.dot-academic { background: #7C93FF; }
.dot-fee { background: #3FD1B5; }

/* ---------- Answers: the left edge shows which document was used ---------- */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    --accent: #8791A5;
    background: rgba(127,127,127,0.06);
    border: 1px solid rgba(127,127,127,0.22);
    border-left: 3px solid var(--accent);
    border-radius: 4px 10px 10px 4px;
    padding: 0.9rem 1.1rem;
}
[data-testid="stChatMessage"]:has(.tag-academic) { --accent: #4F6BED; }
[data-testid="stChatMessage"]:has(.tag-fee)      { --accent: #14A38B; }
[data-testid="stChatMessage"]:has(.tag-error)    { --accent: #E5484D; }

.tag {
    display: inline-block;
    padding: 0.12rem 0.7rem;
    border-radius: 999px;
    font-size: 0.78rem; font-weight: 600;
    margin-bottom: 0.5rem;
}
.tag-academic { background: rgba(79,107,237,0.15);  color: #4F6BED; }
.tag-fee      { background: rgba(20,163,139,0.16);  color: #14A38B; }
.tag-general  { background: rgba(135,145,165,0.20); color: #8791A5; }
.tag-error    { background: rgba(229,72,77,0.15);   color: #E5484D; }

[data-testid="stText"] { white-space: pre-wrap; font-size: 0.85rem; }

/* ---------- Empty state ---------- */
.lead {
    font-family: 'Newsreader', Georgia, serif;
    font-size: 1.55rem; font-weight: 500;
    margin: 0.4rem 0 0.2rem 0;
}
.lead-sub { opacity: 0.7; font-size: 0.92rem; margin-bottom: 1.1rem; }
.group-title { font-weight: 600; font-size: 0.9rem; margin: 0.3rem 0 0.5rem 0; }

/* ---------- Buttons ---------- */
div.stButton > button {
    width: 100%;
    justify-content: flex-start;
    text-align: left;
    border-radius: 6px;
    border: 1px solid rgba(127,127,127,0.30);
    background: transparent;
    padding: 0.6rem 0.85rem;
    font-weight: 500;
    transition: border-color 0.15s ease, transform 0.15s ease;
}
div.stButton > button p { text-align: left; }
div.stButton > button:hover {
    border-color: #4F6BED;
    color: #4F6BED;
    transform: translateX(2px);
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] h3 { font-size: 1rem; margin: 0 0 0.3rem 0; }
.doc-row { display: flex; align-items: center; gap: 0.55rem; font-size: 0.88rem; margin: 0.35rem 0; }
.doc-dot { width: 8px; height: 8px; border-radius: 50%; }
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# YOUR CODE (only changes: the cached loaders and friendly start-up errors)
# -----------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading the language embeddings...")
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embeddings = load_embeddings()

def build_retriver(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    document = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=800,chunk_overlap=100)
    chunks = splitter.split_documents(document)
    
    vectorstore = FAISS.from_documents(chunks,embeddings)
    
    return vectorstore.as_retriever(search_kwargs = {"k":4})


@st.cache_resource(show_spinner="Reading the college documents (first start only)...")
def load_retrievers():
    return (
        build_retriver("academics_handbook.pdf"),
        build_retriver("IITM_Fees_Structure_SAMPLE.pdf"),
    )

try:
    acedemic_retriever, fee_retriever = load_retrievers()
except Exception as e:
    st.error(
        "The college documents could not be loaded. Make sure "
        "`academics_handbook.pdf` and `IITM_Fees_Structure_SAMPLE.pdf` are in the "
        f"same folder as this file.\n\nDetails: `{e}`"
    )
    st.stop()


try:
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.4,
    )
except Exception as e:
    st.error(
        "The language model could not be started. Add `GROQ_API_KEY` to your "
        f".env file and restart the app.\n\nDetails: `{e}`"
    )
    st.stop()

class State(TypedDict):
    programme : str
    messages: Annotated[list,add_messages]
    query_type:str
    retrieved_context:str


def classifier_node(state : State) -> dict:
    """Look at the latest user message and decide which path to take."""

    last_message = state['messages'][-1].content

    prompt = (
        "Classify the following student query into exactly one category: "
        "'academic', 'fee', or 'general'.\n\n"
        "Use 'academic' for questions about attendance, exams, grading, credits, "
        "promotion, course structure, summer training, or degree requirements.\n"
        "Use 'fee' for questions about tuition, payment, refund, late charges, "
        "scholarships, or any money-related topic.\n"
        "Use 'general' for greetings, casual talk, or anything not related to "
        "the college rules or fee.\n\n"
        f"Query: {last_message}\n\n"
        "Return only one word: academic, fee, or general."
    )

    response = llm.invoke(prompt)
    category = response.content.strip().lower()

    if "academic" in category:
        category = "academic"
    elif "fee" in category:
        category = "fee"
    else:
        category = "general"
    
    return {"query_type" : category}


def academic_rag_node(state: State) -> dict:
    """Retrieves relevant chunks from the academics handbook."""
    query = state["messages"][-1].content
    docs = acedemic_retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"retrieved_context": context}

def fee_rag_node(state: State) -> dict:
    """Retrieves relevant chunks from the fee structure PDF."""
    query = state["messages"][-1].content
    docs = fee_retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"retrieved_context": context}


def general_node(state: State) -> dict:
    """Answers directly using the LLM's own knowledge, no retrieval needed."""
    return {"retrieved_context": "NO_RETRIEVAL_NEEDED"}


def response_node(state: State) -> dict:
    """Generates the final answer, personalized using the student's programme."""
    query = state["messages"][-1].content
    programme = state.get("programme", "Unknown")
    context = state["retrieved_context"]

    if context == "NO_RETRIEVAL_NEEDED":
        prompt = (
            f"You are a friendly college assistant talking to a {programme} student. "
            f"Answer this question using your own general knowledge:\n\n{query}"
        )
    else:
        prompt = (
            f"You are a college assistant helping a {programme} student. "
            f"Use the following context from the official college documents to answer "
            f"the question accurately. If the context mentions specific figures for "
            f"different programmes, highlight the one relevant to {programme} if possible.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Give a clear, friendly, and precise answer."
        )

    response = llm.invoke(prompt)
    return {"messages": [("ai", response.content.strip())]}




def route_query(state:State):
    if state['query_type'] == 'academic':
        return "academic_rag"
    elif state['query_type'] == "fee":
        return "fee_rag"
    else:
        return "general"


graph = StateGraph(State)

graph.add_node("classifier",classifier_node)
graph.add_node("academic_rag",academic_rag_node)
graph.add_node("fee_rag",fee_rag_node)
graph.add_node("general",general_node)
graph.add_node("response",response_node)

graph.add_edge(START,"classifier")
graph.add_conditional_edges(
    "classifier",route_query
)
graph.add_edge("academic_rag","response")
graph.add_edge("fee_rag","response")
graph.add_edge("general","response")

graph.add_edge("response",END)

app = graph.compile()


# -----------------------------------------------------------------------------
# STREAMLIT UI (replaces the terminal input loop)
# -----------------------------------------------------------------------------
PROGRAMMES = ["BCA", "BBA", "B.Com (H)"]
USER_AVATAR = "🧑‍🎓"
BOT_AVATAR = "🎓"

CATEGORY = {
    "academic": {"label": "Academic rules", "source": "academics handbook",
                 "step": "Searching the academics handbook"},
    "fee":      {"label": "Fees", "source": "fee structure",
                 "step": "Searching the fee structure"},
    "general":  {"label": "General question", "source": None,
                 "step": "Preparing a general answer"},
    "error":    {"label": "Could not answer", "source": None, "step": ""},
}

SUGGESTIONS = {
    "Academic rules": [
        "What is the minimum attendance I need to sit exams?",
        "How are grades and credits calculated?",
    ],
    "Fees": [
        "What is the tuition fee for my programme?",
        "What are the late payment charges?",
    ],
}

if "messages" not in st.session_state:
    st.session_state.messages = []


def tag_html(category: str, programme: str | None = None) -> str:
    meta = CATEGORY.get(category, CATEGORY["general"])
    text = meta["label"]
    if meta["source"] and programme:
        text += f" for {programme}"
    return f'<span class="tag tag-{category}">{text}</span>'


def show_sources(category: str, context: str | None) -> None:
    source = CATEGORY.get(category, {}).get("source")
    if source and context and context != "NO_RETRIEVAL_NEEDED":
        with st.expander(f"Passages from the {source}"):
            st.text(context)


def render_message(msg: dict) -> None:
    if msg["role"] == "user":
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(msg["content"])
        return
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        st.markdown(tag_html(msg["category"], msg.get("programme")), unsafe_allow_html=True)
        st.markdown(msg["content"])
        show_sources(msg["category"], msg.get("context"))


# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("### Your programme")
    programme = st.radio(
        "Programme", PROGRAMMES, index=0, label_visibility="collapsed"
    )
    st.caption("Answers highlight the figures for this programme when the documents list them.")

    st.divider()
    st.markdown("### Documents in use")
    st.markdown(
        '<div class="doc-row"><span class="doc-dot" style="background:#4F6BED"></span>Academics handbook</div>'
        '<div class="doc-row"><span class="doc-dot" style="background:#14A38B"></span>Fee structure</div>',
        unsafe_allow_html=True,
    )

    st.divider()
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()


# ---------------- Header ----------------
st.markdown(
    f"""
<div class="masthead">
    <div class="masthead-title">College Assistant</div>
    <div class="masthead-sub">
        Ask about attendance, exams, grading, fees and refunds. Every answer is
        drawn from the official college documents.
    </div>
    <div class="masthead-meta">
        <span class="pill pill-prog">Answering for {programme}</span>
        <span class="pill"><i class="dot dot-academic"></i>Academics handbook</span>
        <span class="pill"><i class="dot dot-fee"></i>Fee structure</span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------- Chat ----------------
typed = st.chat_input("Ask about attendance, exams, fees or refunds")
question = typed or st.session_state.pop("pending", None)

for message in st.session_state.messages:
    render_message(message)

if not st.session_state.messages and not question:
    st.markdown('<div class="lead">What would you like to know?</div>', unsafe_allow_html=True)
    st.markdown('<div class="lead-sub">Pick a question to start, or type your own below.</div>', unsafe_allow_html=True)
    cols = st.columns(2)
    for col, (group, items) in zip(cols, SUGGESTIONS.items()):
        with col:
            st.markdown(f'<div class="group-title">{group}</div>', unsafe_allow_html=True)
            for i, text in enumerate(items):
                if st.button(text, key=f"suggestion_{group}_{i}"):
                    st.session_state.pending = text
                    st.rerun()

if question:
    user_message = {"role": "user", "content": question}
    render_message(user_message)

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        tag_slot = st.empty()
        status = st.status("Reading your question", expanded=False)
        text_slot = st.empty()

        final, error = None, None
        category = "general"
        try:
            # Same graph, same input as before - streamed so the UI can show progress.
            for step in app.stream(
                {"programme": programme, "messages": [("human", question)]},
                stream_mode="values",
            ):
                final = step
                query_type = step.get("query_type")
                ctx = step.get("retrieved_context")

                if query_type and not ctx:
                    category = query_type
                    tag_slot.markdown(tag_html(category, programme), unsafe_allow_html=True)
                    status.write(f"Understood as: {CATEGORY[category]['label'].lower()}")
                    status.update(label=CATEGORY[category]["step"])
                elif ctx and len(step["messages"]) == 1:
                    if ctx != "NO_RETRIEVAL_NEEDED":
                        status.write(f"Found relevant passages in the {CATEGORY[category]['source']}")
                    status.update(label="Writing your answer")

            answer = final["messages"][-1].content
            context = final.get("retrieved_context")
        except Exception as e:
            error = e

        if error is None:
            source = CATEGORY[category]["source"]
            status.update(
                label=f"Answered from the {source}" if source else "Answered from general knowledge",
                state="complete",
                expanded=False,
            )
            # Reveal the answer a few words at a time
            words = answer.split(" ")
            if len(words) <= 400:
                for i in range(0, len(words), 3):
                    text_slot.markdown(" ".join(words[: i + 3]) + " ▌")
                    time.sleep(0.02)
            text_slot.markdown(answer)
            show_sources(category, context)
            reply = {"role": "assistant", "content": answer, "category": category,
                     "context": context, "programme": programme}
        else:
            category = "error"
            tag_slot.markdown(tag_html("error"), unsafe_allow_html=True)
            status.update(label="Something went wrong", state="error", expanded=False)
            answer = (
                "No answer could be produced. Check that `GROQ_API_KEY` is set in your "
                f".env file and that you are online, then ask again.\n\nDetails: `{error}`"
            )
            text_slot.markdown(answer)
            reply = {"role": "assistant", "content": answer, "category": "error",
                     "context": None, "programme": programme}

    st.session_state.messages.append(user_message)
    st.session_state.messages.append(reply)