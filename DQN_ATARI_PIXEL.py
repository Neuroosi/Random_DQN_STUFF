import torch
from torch import nn
from torch._C import device
from torch import optim
import random
import numpy as np
from collections import deque
from collections import deque
import flappy_bird_gym as fg
import skimage
from graphs import graph
import gym
##HYPERPARAMETERS
learning_rate = 0.00015
REPLAY_MEMORY_SIZE=10**4
BATCH_SIZE = 64
GAMMA = 0.99
EPISODES = 5000
EPSILON = 1
EPSILON_MIN = 0.1
EPSILON_DECAY = (EPSILON-EPSILON_MIN)/30000
INPUTSIZE = (84,84)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
torch.autograd.set_detect_anomaly(True)

class NeuralNetwork(nn.Module):
    def __init__(self, actionSpaceSize, obsSpaceSize):
        self.actionSpaceSize = actionSpaceSize
        self.obsSpaceSize = obsSpaceSize
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, 8, 4)
        torch.nn.init.kaiming_uniform_(self.conv1.weight)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        torch.nn.init.kaiming_uniform_(self.conv2.weight)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        torch.nn.init.kaiming_uniform_(self.conv3.weight)

        self.fc1 = nn.Linear(3136, 256)
        torch.nn.init.kaiming_uniform_(self.fc1.weight)
        # Output 2 values: fly up and do nothing
        self.fc2 = nn.Linear(256, self.actionSpaceSize)
        torch.nn.init.kaiming_uniform_(self.fc2.weight)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.to(device)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        # Flatten output to feed into fully connected layers
        x = x.view(x.size()[0], -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


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
        states = states.float()
        next_states = [torch.from_numpy(np.array(transition[3])) for transition in batch]
        next_states = torch.stack(next_states)
        next_states = next_states.float()
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
                state = np.expand_dims(state, axis=0)
                state = torch.from_numpy(state)
                state = state.float()
                return torch.argmax(model(state)).item()
        return random.randrange(model.actionSpaceSize)

    def saveModel(self, agent):
        torch.save(agent.state_dict(), 'pixel_atari_weights.pth')
        print("Model saved!")
    def loadModel(self, agent):
        agent.load_state_dict(torch.load("pixel_atari_weights.pth"))
        print("Model loaded!")
def getFrame(x):
    state = skimage.color.rgb2gray(x)
    state = skimage.transform.resize(state, INPUTSIZE)
    state = skimage.exposure.rescale_intensity(state,out_range=(0,255))
    return state/255

def makeState(state):
    return np.stack((state[0],state[1],state[2],state[3]), axis=0)

if __name__ == "__main__":
    env = gym.make('BreakoutDeterministic-v4')
    y = NeuralNetwork(env.action_space.n, None).to(device)
    target_y = NeuralNetwork(env.action_space.n, None).to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(y.parameters(), lr = learning_rate)
    agent = DQN()
    state = deque(maxlen = 4)
    print(y)
    answer = input("Use a pre-trained model y/n? ")
    if answer == "y":
        agent.loadModel(y)
        agent.loadModel(target_y)
        EPSILON = 0.5
    t = 0
    rewards = []
    avgrewards = []
    dist = 0
    for episode in range(1,EPISODES+500000000000):
        obs = env.reset()
        cumureward = 0
        state.append(getFrame(obs))
        state.append(getFrame(obs))
        state.append(getFrame(obs))
        state.append(getFrame(obs))
        while True:
            action = agent.getPrediction(makeState(state),y)
            obs, reward, done, info = env.step(action)
            env.render()
            cache = state.copy()
            state.append(getFrame(obs))
            agent.update_replay_memory((makeState(cache), action, reward, makeState(state), done))
            if len(agent.replay_memory) > 2000:
                agent.train(y, target_y, loss_fn, optimizer)
                EPSILON = max(EPSILON_MIN, EPSILON-EPSILON_DECAY)
                if t % 1000 == 0:
                    target_y.load_state_dict(y.state_dict())
            t+=1
            cumureward += reward
            if t % 10000 == 0:
                agent.saveModel(y)
                graph(rewards, avgrewards,"fetajuusto/DQN-FLAPPY-PIXEL")
            if done or info["ale.lives"] != 5:
                break
        rewards.append(cumureward)
        avgrewards.append(np.sum(np.array(rewards))/episode)
        print("Score:", cumureward, " Episode:", episode, "Time:", t , " Epsilon:", EPSILON)
