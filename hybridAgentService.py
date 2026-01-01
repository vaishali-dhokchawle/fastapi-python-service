import pandas as pd
import joblib as jb
from agentic import calculateReward
from agentic import HybridRLAgent

class HybridAgentService:
 def __init__(self):
     self.lr_model=jb.load("bms_lr_model.pkl")
     self.hybridAgent=HybridRLAgent()
     self.prev_state=None

 def process(self,state):
    X=pd.DataFrame([state])
    predicted_temp=self.lr_model.predict(X)[0]
    lr_state={"temperature":state["temperature"],
              "predicted_temp":predicted_temp,
              "occupancy":state["occupancy"]}
    action=self.hybridAgent.choose_action(lr_state)

    reward=0
    if self.prev_state :
       reward=calculateReward(state["temperature"],action)
       self.hybridAgent.learn(self.prev_state,action,reward,lr_state)

    self.prev_state=lr_state

    return{
       "predicted_temp":predicted_temp,
       "action":action
    }

