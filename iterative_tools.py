import os
import json

from typing import TypedDict, Annotated

import serpapi
from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_core.messages import ToolMessage

from langchain_huggingface import (
    HuggingFaceEndpoint,
    ChatHuggingFace,
)

from langchain_groq import ChatGroq

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode



load_dotenv()
HF_TOKEN = os.getenv("HF_API_KEY")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in .env")


SERPAPI_KEY = os.getenv("SEARCH_API_KEY")

if not SERPAPI_KEY:
    raise ValueError(
        "SEARCH_API_KEY not found in environment variables."
    )


@tool
def web_search(query: str) -> str:
    """
    Search Google for current information.

    Use this tool when the topic requires:
    - current news
    - latest statistics
    - recent trends
    - current events
    - up-to-date information
    """

    try:
        client = serpapi.Client(
            api_key=SERPAPI_KEY
        )

        results = client.search(
            {
                "engine": "google",
                "q": query,
                "num": 3,
            }
        )

        return json.dumps(results, indent=2, default=str)

    except Exception as e:
        return f"Search failed: {str(e)}"


tools = [web_search]
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4.1-Flash",
    max_new_tokens=300,
    provider="fireworks-ai",
    huggingfacehub_api_token=HF_TOKEN,
)

writer_llm = ChatHuggingFace(
    llm=llm
)

writer_llm_with_tools = writer_llm.bind_tools(tools)



reviewer_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.1,
)




class State(TypedDict):
    topic: str

    messages: Annotated[
        list,
        add_messages
    ]

    draft: str
    review_feedback: str
    is_approved: bool
    attempt: int




WRITER_SYSTEM_PROMPT = """
You are an expert LinkedIn content writer.

Your job is to write engaging, professional LinkedIn posts
about the given topic.

If the topic requires up-to-date information, statistics,
current news, or recent trends, use the web_search tool
before writing the post.

Important:
- If you use the web search tool, carefully read the returned
  search results before writing the final post.
- Do not mention the search process in the LinkedIn post.
- Do not invent statistics or facts.
- Use only information supported by the search results.

If you have already received reviewer feedback on a previous
draft, carefully address every issue mentioned in the feedback.

Rules for good LinkedIn posts:

1. Strong hook in the first line.
2. One clear and valuable takeaway.
3. Easy to skim.
4. Use short paragraphs.
5. Around 150-200 words.
6. Professional but human tone.
7. Avoid corporate-robotic language.
8. End with a question or call-to-action.
9. Do not use hashtags.
10. Do not add unnecessary headings.
"""



def writer_node(state: State) -> dict:
    """
    Writes the LinkedIn post.

    There are three possible situations:

    1. First call:
       Generate the first draft.

    2. Tool result available:
       Read the tool result and continue writing.

    3. Reviewer rejected previous draft:
       Generate a new improved draft.
    """

    topic = state["topic"]

    messages = state.get("messages", [])

    attempt = state.get("attempt", 0)

    feedback = state.get(
        "review_feedback",
        ""
    )



    if not messages:

        user_message = (
            f"Write a LinkedIn post about this topic:\n\n"
            f"{topic}\n\n"
            f"If current information is required, "
            f"use the web_search tool first."
        )

        response = writer_llm_with_tools.invoke(
            [
                ("system", WRITER_SYSTEM_PROMPT),
                ("human", user_message),
            ]
        )

        return {
            "messages": [response],

            "attempt": 1,
        }



    last_message = messages[-1]

    if isinstance(last_message, ToolMessage):

        response = writer_llm_with_tools.invoke(
            [
                ("system", WRITER_SYSTEM_PROMPT),
                *messages,
            ]
        )


        return {
            "messages": [response],
        }


    user_message = (
        f"Your previous LinkedIn post about "
        f"'{topic}' was rejected by the reviewer.\n\n"

        f"Here is the reviewer's feedback:\n\n"
        f"{feedback}\n\n"

        f"Write a completely improved version of the post.\n\n"

        f"Make sure you fix EVERY issue mentioned by "
        f"the reviewer.\n"

        f"Do not repeat the same mistakes."
    )

    response = writer_llm_with_tools.invoke(
        [
            ("system", WRITER_SYSTEM_PROMPT),
            ("human", user_message),
        ]
    )

    return {
        "messages": [response],

        # New actual writing attempt
        "attempt": attempt + 1,
    }




tool_node = ToolNode(tools)



def extract_draft_node(state: State) -> dict:
    """
    Extracts the final AI-generated LinkedIn post
    from the last AI message.
    """

    messages = state["messages"]

    last_message = messages[-1]

    draft = last_message.content


    if isinstance(draft, list):

        text_parts = []

        for block in draft:

            if isinstance(block, dict):

                if block.get("type") == "text":
                    text_parts.append(
                        block.get("text", "")
                    )

            else:
                text_parts.append(str(block))

        draft = "\n".join(text_parts)

    draft = str(draft).strip()

    print("\n")
    print("=" * 55)
    print("GENERATED DRAFT")
    print("=" * 55)
    print(draft)
    print("=" * 55)

    return {
        "draft": draft
    }



REVIEWER_SYSTEM_PROMPT = """
You are a strict LinkedIn content reviewer.

Your job is to decide whether a LinkedIn post is
publish-ready.

Evaluate the post against ALL of these criteria:

1. Strong hook in the first line.
2. One clear and valuable takeaway.
3. Easy to skim using short paragraphs.
4. Roughly 150-200 words.
5. Ends with an engaging question or CTA.
6. Professional but human tone.
7. Not corporate or robotic.
8. No hashtags.

Be strict but fair.

Approve ONLY when the post genuinely satisfies
all important criteria.

If even one criterion is clearly missing,
reject the post.

Respond in EXACTLY this format:

VERDICT: APPROVED
FEEDBACK: <one short paragraph>

OR

VERDICT: REJECTED
FEEDBACK: <one short paragraph>
"""


def reviewer_node(state: State) -> dict:
    """
    Reviews the generated LinkedIn post.
    """

    draft = state["draft"]

    prompt = (
        "Review the following LinkedIn post draft.\n\n"
        "----- START DRAFT -----\n"
        f"{draft}\n"
        "----- END DRAFT -----\n\n"
        "Evaluate it against every criterion."
    )

    response = reviewer_llm.invoke(
        [
            (
                "system",
                REVIEWER_SYSTEM_PROMPT
            ),
            (
                "human",
                prompt
            ),
        ]
    )

    review_text = response.content.strip()

 

    verdict_section = review_text.upper().split(
        "FEEDBACK:",
        1
    )[0]

    is_approved = (
        "VERDICT: APPROVED" in verdict_section
    )


    if "FEEDBACK:" in review_text:

        feedback = review_text.split(
            "FEEDBACK:",
            1
        )[1].strip()

    else:

        feedback = review_text

    verdict = (
        "APPROVED"
        if is_approved
        else "REJECTED"
    )

    print("\n")
    print("=" * 55)
    print(f"REVIEWER VERDICT: {verdict}")
    print("=" * 55)
    print(f"Feedback: {feedback}")
    print("=" * 55)

    return {
        "review_feedback": feedback,
        "is_approved": is_approved,
    }




def should_use_tool(state: State):
    """
    Decide whether the writer wants to call a tool.

    If the last AI message contains tool calls:
        writer → tools

    Otherwise:
        writer → extract_draft
    """

    last_message = state["messages"][-1]

    tool_calls = getattr(
        last_message,
        "tool_calls",
        None
    )

    if tool_calls:
        return "tools"

    return "extract_draft"



def should_stop_looping(state: State):
    """
    Decide whether the workflow should stop or
    send the post back to the writer.
    """
    if state["is_approved"]:
        print("\nPost has been approved.")
        return END
    if state["attempt"] >= 3:
        print("\nReached maximum attempts.")
        return END
    return "writer"

graph = StateGraph(State)
graph.add_node(
    "writer",
    writer_node
)
graph.add_node(
    "tools",
    tool_node
)
graph.add_node(
    "extract_draft",
    extract_draft_node
)
graph.add_node(
    "reviewer",
    reviewer_node
)
graph.add_edge(
    START,
    "writer"
)
graph.add_conditional_edges(
    "writer",
    should_use_tool,
)
graph.add_edge(
    "tools",
    "writer"
)
graph.add_edge(
    "extract_draft",
    "reviewer"
)
graph.add_conditional_edges(
    "reviewer",
    should_stop_looping
)

app = graph.compile()

print("=" * 55)
print("       LINKEDIN MULTI-AGENT CONTENT GENERATOR")
print("=" * 55)

print("\nThis system will:")
print(
    "1. Generate a LinkedIn post"
)

print(
    "2. Search the web when current information is required"
)

print(
    "3. Review the generated post"
)

print(
    "4. Rewrite the post if rejected"
)

print(
    "5. Stop after approval or 3 writing attempts"
)

print("=" * 55)



topic = input(
    "\nWhat topic do you want a LinkedIn post about?\n> "
).strip()



if not topic:

    print(
        "\nNo topic was provided. Exiting."
    )

else:

    print(
        "\nStarting generation..."
    )


    initial_state: State = {

        "topic": topic,

        "messages": [],

        "draft": "",

        "review_feedback": "",

        "is_approved": False,

        "attempt": 0,
    }


    final_state = app.invoke(
        initial_state
    )

 

    print("\n")
    print("=" * 55)
    print("              FINAL LINKEDIN POST")
    print("=" * 55)

    print(
        final_state["draft"]
    )

    print("=" * 55)

    print(
        f"Total writing attempts: "
        f"{final_state['attempt']}"
    )

    print(
        f"Approved: "
        f"{final_state['is_approved']}"
    )

    print("=" * 55)