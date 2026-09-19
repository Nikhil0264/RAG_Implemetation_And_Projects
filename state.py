import os

from typing import TypedDict
class State(TypedDict):
    topic:str
    summary:str
    score:int
    

from pydantic import BaseModel,field_validator  
class State(BaseModel):
    topic:str 
    summary:str =""
    score:int
    
    @field_validator
    def score_positive(cls,v):
        if v < 0:
            raise ValueError("Score must be positive")
        return v
    

