from langchain_groq import ChatGroq
import os
from typing import TypedDict,Annotated
from langgraph.graph import StateGraph,START,END
from dotenv import load_dotenv
load_dotenv()

groq_key = os.getenv("GROQ_KEY")
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.1,
)

def merge_score_dicts(existing:dict,newupdate :dict)->dict:
    if existing is None:
        return newupdate
    return {**existing,**newupdate}

class AnalyzerState(TypedDict):
    raw_text : str
    safety_scores : Annotated[dict[str,int],merge_score_dicts]
    
    
def toxicity_node(state: AnalyzerState) -> dict:
    print("\n [Branch 1] Analyzing Toxicity and Hate Speech...")
    prompt = (
        "Analyze the following text for profanity, aggression, hate speech, or toxicity. "
        "Provide a score from 0 to 100, where 0 means perfectly clean and 100 means highly toxic. "
        "Return ONLY the plain integer number, nothing else.\n\n"
        f"Text:\n{state['raw_text']}"
    )
    response = llm.invoke(prompt)
    try:
        score = int(response.content.strip())
    except ValueError:
        score = 0
        
    return {"safety_scores": {"toxicity_level": score}}

def copyright_node(state: AnalyzerState) -> dict:
    print("\n🔏 [Branch 2] Analyzing Copyright & Originality Risks...")
    prompt = (
        "Analyze the following text. Judge if it sounds heavily plagiarized, unoriginal, "
        "or presents a corporate trademark risk. Provide a score from 0 to 100, "
        "where 0 means entirely original and 100 means high risk. "
        "Return ONLY the plain integer number, nothing else.\n\n"
        f"Text:\n{state['raw_text']}"
    )
    response = llm.invoke(prompt)
    try:
        score = int(response.content.strip())
    except ValueError:
        score = 0
        
    
    return {"safety_scores": {"copyright_risk": score}}


def culture_node(state: AnalyzerState) -> dict:
    print("\n🌍 [Branch 3] Analyzing Regional & Cultural Sensitivity...")
    prompt = (
        "Analyze the following text for regional sensitivities, political landmines, "
        "or cultural insensitivity that might offend a global audience. Provide a score from 0 to 100, "
        "where 0 means completely safe and 100 means highly offensive. "
        "Return ONLY the plain integer number, nothing else.\n\n"
        f"Text:\n{state['raw_text']}"
    )
    response = llm.invoke(prompt)
    try:
        score = int(response.content.strip())
    except ValueError:
        score = 0
        
    
    return {"safety_scores": {"cultural_insensitivity": score}}


builder = StateGraph(AnalyzerState)

builder.add_node("toxicity_node",toxicity_node)
builder.add_node("copyright_check",copyright_node)
builder.add_node("cultural_node",culture_node)

builder.add_edge(START,"toxicity_node")
builder.add_edge(START,"copyright_check")
builder.add_edge(START,"cultural_node")

builder.add_edge("toxicity_node",END)
builder.add_edge("copyright_check",END)
builder.add_edge("cultural_node",END)

app = builder.compile()

sample_script = """
   Today we're going to talk about how AI is changing the way we use smartphones. Most people think AI is only about chatbots or image generation, but that's actually just one small part of the story.
    Your phone is already using AI in the background. For example, when your camera automatically detects a face, improves low-light photos, or removes unwanted objects from an image, AI models are doing a lot of the processing.
    Another interesting example is voice recognition. When you ask your phone to set a reminder or send a message, the system first understands what you're saying and then converts that request into an action.
    And this technology is getting more powerful every year. Newer smartphones can run smaller AI models directly on the device, which means some tasks don't even need an internet connection.
    So the next time your phone seems to magically understand what you want, remember that there's actually a lot of AI working behind the scenes.
    """
    

    
initial_state = {
    "raw_text": sample_script,
    "safety_scores": {} 
}
    
final_state = app.invoke(initial_state)
    

print(final_state["safety_scores"])