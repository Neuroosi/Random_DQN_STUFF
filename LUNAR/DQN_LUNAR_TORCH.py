import torch
from torch import nn
from torch._C import device
from torch import optim
import random
import numpy as np
from collections import deque
import gym
import FFNN

##HYPERPARAMETERS
learning_rate = 0.00001
REPLAY_MEMORY_SIZE=100000
BATCH_SIZE = 64
GAMMA = 0.99
EPISODES = 5000
EPSILON = 1
EPSILON_MIN = 0.1
EPSILON_DECAY = (EPSILON-EPSILON_MIN)/30000
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
torch.autograd.set_detect_anomaly(True)


class DQN(nn.Module):

    def __init__(self):
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.ddqn = True

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def train(self, agent, target, loss_fn, optimizer):
        batch = random.sample(self.replay_memory, BATCH_SIZE)
        X = []
        Y = []
        states = [torch.from_numpy(np.array(transition[0])) for transition in batch]
        states = torch.stack(states)
        next_states = [torch.from_numpy(np.array(transition[3])) for transition in batch]
        next_states = torch.stack(next_states)
        optimizer.zero_grad()
        y = agent(states)
        target_y = target(next_states)
        y_next = agent(next_states)
        for i,(state, action, reward, next_state, done) in enumerate(batch):
            if done:
                y[i][action] = reward
            elif self.ddqn is False:
                y[i][action] = reward + GAMMA*torch.max(target_y[i])
            else:
                y[i][action] = reward + GAMMA*target_y[i][torch.argmax(y_next[i])]
            X.append(torch.from_numpy(state))
            Y.append(y[i])
        Y = torch.stack(Y)
        agent.train()
        pred = agent(states)
        loss = loss_fn(pred, Y)
        loss.backward()
        optimizer.step()      

    def getPrediction(self, state, model):
        if np.random.rand() > EPSILON:
            with torch.no_grad():
                return torch.argmax(model(torch.from_numpy(np.array([state])))).item()
        return random.randrange(model.actionSpaceSize)

    def saveModel(self, agent):
        torch.save(agent.state_dict(), 'model_weights.pth')
        print("Model saved!")
    def loadModel(self, agent):
        agent.load_state_dict(torch.load("model_weights.pth"))
        print("Model loaded!")

if __name__ == "__main__":
    env = gym.make('LunarLander-v2')
    env.frameskip = 4
    env.configure(batch_mode = True)
    state = env.reset()
    y = FFNN.NeuralNetwork(env.action_space.n, state.shape[0]).to(device)
    target_y = FFNN.NeuralNetwork(env.action_space.n, state.shape[0]).to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(y.parameters(), lr = learning_rate)
    agent = DQN()
    answer = input("Use a pre-trained model y/n? ")
    if answer == "y":
        agent.loadModel(y)
        agent.loadModel(target_y)
        EPSILON = 0.1
    t = 0
    rewards = []
    cumureward = 0
    for episode in range(EPISODES+500000000000):
        state = env.reset()
        cumureward = 0
        while True:
            env.render()
            cache = state.copy()
            action = agent.getPrediction(state,y)
            state, reward, done, info = env.step(action)
            agent.update_replay_memory((cache, action, reward, state, done))
            if len(agent.replay_memory) > 10000:
                agent.train(y, target_y, loss_fn, optimizer)
                EPSILON = max(EPSILON_MIN, EPSILON-EPSILON_DECAY)
                if t % 1000 == 0:
                    target_y.load_state_dict(y.state_dict())
            t+=1
            cumureward += reward
            if t % 10000 == 0:
                agent.saveModel(y)
            if done:
                break
        print("Score:", cumureward, " Episode:", episode, "Time:", t , " Epsilon:", EPSILON)