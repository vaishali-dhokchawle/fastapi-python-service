from fastapi import FastAPI
from hybridAgentService import HybridAgentService

api=FastAPI()
bmsService=HybridAgentService()

@api.post("/decide")
def decide (state:dict):
    return bmsService.process(state)