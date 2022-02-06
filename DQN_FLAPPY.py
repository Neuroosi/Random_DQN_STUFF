import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Lambda, Compose
from tensorflow import keras
import random
import numpy as np
from collections import deque
import gym

##HYPERPARAMETERS
learning_rate = 0.0001
REPLAY_MEMORY_SIZE=100000
BATCH_SIZE = 64
GAMMA = 0.99
EPISODES = 5000
EPSILON = 1
EPSILON_MIN = 0.1
EPSILON_DECAY = (EPSILON-EPSILON_MIN)/30000

class DQN(nn.Module):

    def __init__(self, actionSpaceSize, obsSpaceSize):
        self.actionSpaceSize = actionSpaceSize
        self.obsSpaceSize = obsSpaceSize
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.ddqn = True
        super(DQN, self).__init__()
        self.flatten = nn.Flatten()
        self.Q = nn.Sequential(nn.Linear(self.obsSpaceSize, 64),
        nn.ReLU(),
        nn.Linear(64, self.actionSpaceSize))
        self.Q_target = nn.Sequential(nn.Linear(self.obsSpaceSize, 64),
        nn.ReLU(),
        nn.Linear(64, self.actionSpaceSize))

    def forward_Q(self, x):
        x = self.flatten(x)
        Q_values = self.Q(x)
        return Q_values 

    def forward_Q_target(self,x):
        x = self.flatten(x)
        Q_values = self.Q_target(x)
        return Q_values

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def train(self):
        batch = random.sample(self.replay_memory, BATCH_SIZE)
        next_states = [transition[3] for transition in batch]
        X = []
        Y = []
        y = [self.forward_Q(np.array(transition[0])) for transition in batch]
        target_y = [self.forward_Q_target(np.array(transition[3])) for transition in batch]
        y_next = [self.forward_Q(np.array(transition[3])) for transition in batch]
        for i,(state, action, reward, next_state, done) in enumerate(batch):
            if done:
                y[i][action] = reward
            elif self.ddqn is False:
                y[i][action] = reward + GAMMA*np.amax(target_y[i])
            else:
                y[i][action] = reward + GAMMA*target_y[i][np.argmax(y_next[i])]
            X.append(state)
            Y.append(y[i])
        self.model.train_on_batch(np.array(X), np.array(Y))

    def updateTargetNetwork(self):
        self.target_model.set_weights(self.model.get_weights())            

    def getPrediction(self, state):
        if np.random.rand() > EPSILON:
            return np.argmax(self.model.predict(np.array([state])))
        return random.randrange(self.actionSpaceSize)

    def saveModel(self):
        self.model.save("/home/joel/Flappybird-DQN/bestmodel")
    def loadModel(self):
        self.model = keras.models.load_model("/home/joel/Flappybird-DQN/bestmodel/")

if __name__ == "__main__":
    env = gym.make('LunarLander-v2')
    state = env.reset()
    agent = DQN(env.action_space.n, state.shape[0])
    answer = input("Use a pre-trained model y/n? ")
    if answer is "y":
        agent.loadModel()
        EPSILON = 0.1
    t = 0
    rewards = []
    episodes = []
    cumureward = 0
    for episode in range(EPISODES+500000000000):
        state = env.reset()
        cumureward = 0
        while True:
            env.render()
            cache = state.copy()
            action = agent.getPrediction(state)
            state, reward, done, info = env.step(action)
            agent.update_replay_memory((cache, action, reward, state, done))
            if len(agent.replay_memory) > 1000:
                agent.train()
                EPSILON = max(EPSILON_MIN, EPSILON-EPSILON_DECAY)
                if t % 1000 == 0:
                    agent.updateTargetNetwork()
            t+=1
            cumureward += reward
            if t % 10000 == 0:
                agent.saveModel()
            if done:
                break
        print("Score:", cumureward, " Episode:", episode, "Time:", t , " Epsilon:", EPSILON)