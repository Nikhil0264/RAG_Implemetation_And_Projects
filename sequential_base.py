import os
from typing import TypedDict

class pipelineState(TypedDict):
    raw_input:str
    edited_text:str
    script_text:str
    final_output:str
    
    
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

groq_key = os.getenv("GROQ_KEY")
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.7,
)

def editor_node(state:pipelineState)->dict:
    """Stage 1: Cleans up grammer,removes typos , and refines the tone"""
    
    prompt = (
        "you are an expert,copyeditor.Clean up the following raw text."
        "Fix any grammatical errors ,spelling mistakes,and smooth out the transition flow"
        "while keeping the core message intact.Return only the edited text.\n\n"
        f"Text:\n{state['raw_input']}"
    )
    
    response = llm.invoke(prompt)
    
    return {"edited_text" : response.content.strip()}


def scriptwriter_node(state:pipelineState)->dict:
    """Stage 2 :Formats the clean text into an enagaging video script style"""
    print("\n---[Stage 2] Executing Scriptwriter Node -----")
    
    prompt = (
        "you are charimatic youtube content creator.Take this edited text and transform"
        "it into highly engagging ,punchy,conversational video script hook.Make it sound"
        "like a real person speaking passionately.Return only the script content.\n\n"
        f"Edited Text :\n{state['edited_text']}"
    )
    
    response = llm.invoke(prompt)
    return {"script_text":response.content.strip()}


def translator_node(state:pipelineState)->dict:
    """Stage 3:Translates the script into natural flowing Hinglish"""
    print("\n---Stage 3 Executing Hinglish Translator Node ---")
    
    prompt = (
        "you are an expert sontent localizer for Indian market.Take the following script"
        "and convert it into natural ,flowing 'Hinglish'.Do not simply translate it sentence-by-"
        "or reapeat information. Altering comfortably between Hindi and English phrases just like"
        "an intellectual tech educator would speak naturally on live stream keep the energy "
        "Return only the final HInglish text.\n\n"
        f"Script :\n{state['script_text']}"
    )

    response = llm.invoke(prompt)
    return {"final_output" : response.content.strip()}



from langgraph.graph import StateGraph,START,END

graph = StateGraph(pipelineState)

graph.add_node("editor",editor_node)
graph.add_node("scripteritor",scriptwriter_node)
graph.add_node("translator",translator_node)

graph.add_edge(START,"editor")
graph.add_edge("editor","scripteritor")
graph.add_edge("scripteritor","translator")
graph.add_edge("translator",END)

app = graph.compile()

result = app.invoke({
    "raw_input":"""Today we're going to talk about how AI is changing the way we use smartphones. Most people think AI is only about chatbots or image generation, but that's actually just one small part of the story.
    Your phone is already using AI in the background. For example, when your camera automatically detects a face, improves low-light photos, or removes unwanted objects from an image, AI models are doing a lot of the processing.
    Another interesting example is voice recognition. When you ask your phone to set a reminder or send a message, the system first understands what you're saying and then converts that request into an action.
    And this technology is getting more powerful every year. Newer smartphones can run smaller AI models directly on the device, which means some tasks don't even need an internet connection.
    So the next time your phone seems to magically understand what you want, remember that there's actually a lot of AI working behind the scenes."""
})

print("\n"+" -"*50)
print(result['final_output'])