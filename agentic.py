import random
import pandas as pd
import numpy as np
import joblib as jb
from sklearn.linear_model import LinearRegression

class RoomTemperature:
    def __init__ (self):
        self.temperature=22
        self.occupancy=1
        self.humidity=55.0
        self.co2=600.0
        self.energy=0.0
        self.hvac_on=0
    def step(self,action):
        self.hvac_on=action
        if self.hvac_on==1:
            self.temperature -= 0.8
            self.energy +=0.5
        else: self.temperature += 3.0

        if self.occupancy==1:
            self.temperature+=0.2
            self.co2+=20
        else: self.co2-=10
 
        self.temperature += random.uniform(-0.3,0.3)
        return { "temperature":self.temperature,
                 "humidity":self.humidity,
                "co2":self.co2,
                "energy":self.energy,
                "hvac_on":self.hvac_on,
                "occupancy":self.occupancy}
    


class BMSAgent:
    def __init__(self,target_temp=22):
        self.target_temp=target_temp

    def decide(self,predict_temp):
        if predict_temp>self.target_temp+1:
            return 1
        elif predict_temp<self.target_temp-1:
            return -1
        else:
            return 0

def calculateReward(temp,action):
    comfort_penalty= abs(temp-22)
    comfort_energy=abs(action)*0.5
    return -(comfort_penalty+comfort_energy)

class HybridRLAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.q = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def to_scalar(self,x):
        if hasattr(x, "iloc"):
            return x.iloc[0]
        return x

    def get_key(self, s):
        return (
            int(round(self.to_scalar(s["temperature"]))),
            int(round(self.to_scalar(s["predicted_temp"]))),
            int(self.to_scalar(s["occupancy"]))
        )


    def choose_action(self, state):
        import random
        if random.random() < self.epsilon:
            return random.choice([0, 1])

        key = self.get_key(state)
        return max([0, 1], key=lambda a: self.q.get((key, a), 0))

    def learn(self, s, a, r, s_next):
        key = self.get_key(s)
        key_next = self.get_key(s_next)

        best_next = max(self.q.get((key_next, a2), 0) for a2 in [0, 1])
        self.q[(key, a)] = self.q.get((key, a), 0) + self.alpha * (
            r + self.gamma * best_next - self.q.get((key, a), 0)
        )

    
env=RoomTemperature()

agent=HybridRLAgent()
lr_model=jb.load("bms_lr_model.pkl")
X_new = pd.DataFrame([{
    "temperature":env.temperature,
    "humidity":env.humidity,
    "co2":env.co2,
    "energy":env.energy,
    "hvac_on":env.hvac_on,
    "occupancy":env.occupancy
}])

for step in range(10):
   
    predicted_temp= lr_model.predict(X_new)[0]
   
    rl_state={"temperature":X_new["temperature"],
                "predicted_temp":predicted_temp,
                "occupancy":X_new["occupancy"]}
    action=agent.choose_action(rl_state)

    new_state=env.step(action)
    new_temp=new_state["temperature"]
    reward=calculateReward(new_temp,action)
    X_next=pd.DataFrame([new_state])
    predicted_next_temp=lr_model.predict(X_next)[0]
    rl_next_state={"temperature":X_next["temperature"],
                "predicted_temp":predicted_next_temp,
                "occupancy":X_next["occupancy"]}
    agent.learn(rl_state,action,reward,rl_next_state)

    print(f"Step {step+1}")
    print(f"Temp: {new_temp:.2f}°C | Action: {action} | Reward: {reward:.2f}")
    print("-" * 40)

    X_new=pd.DataFrame([{"temperature":new_state["temperature"],"humidity":new_state["humidity"],
            "co2":new_state["co2"],"energy":new_state["energy"],
            "hvac_on":new_state["hvac_on"],
            "occupancy":new_state["occupancy"]}])


