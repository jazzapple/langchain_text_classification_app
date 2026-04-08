from langchain_ollama import ChatOllama

from .prompts import classification_prompt
from .schema import ClassificationResult


def build_chain(model: str = "qwen2.5:3b"):
    llm = ChatOllama(model=model)
    structured_llm = llm.with_structured_output(ClassificationResult)
    return classification_prompt | structured_llm


# The | is the LangChain expression language (LCEL) pipe operator. It chains two components together into a Runnable — meaning when you call .invoke(), it passes  
#   the output of the left side as input to the right side.                                                                                                          
                                                                                                                                                                   
#   So classification_prompt | structured_llm means:                                                                                                                                                                    
#   1. classification_prompt.invoke({"text": "..."}) → formats the prompt into a list of chat messages                                                               
#   2. That output is passed directly into structured_llm.invoke(messages) → calls Ollama and returns a ClassificationResult
                                                                                                                                                                   
#   It's equivalent to writing:                                                                                                                                                                              
#   messages = classification_prompt.invoke({"text": text})   
#   result = structured_llm.invoke(messages)                                                                                                                         
   
#   But the pipe lets you compose arbitrarily long chains (e.g. prompt | llm | parser | validator) as a single reusable object, and gives you things like streaming  
#   and batch processing for free across the whole chain.   