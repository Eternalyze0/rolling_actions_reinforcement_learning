# Rolling Actions Reinforcement Learning

Instead of using epsilon-greedy or injecting noise we roll the action indices. Baseline DQN code from https://raw.githubusercontent.com/seungeunrho/minimalRL/refs/heads/master/dqn.py.

## Usage
```
python3.10 rolling_actions_dqn.py
```

## Key Code Changes

```py
        epsilon = 0.0
```
```py
class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)
        self.ri = torch.tensor(0)

    def forward(self, x):
        x = F.relu(self.fc1(x + self.ri))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        if self.ri % 2 == 1:
            x = torch.roll(x, 1)
        self.ri += 1
        self.ri %= 2
        return x
```
Alternatively,
```py
class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(6, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)
        self.ri = torch.tensor(0)

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        x = F.relu(self.fc1(torch.cat([x, F.one_hot(self.ri, num_classes=2).expand(x.shape[0], 2)], dim=1)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        if self.ri % 2 == 1:
            x = torch.roll(x, 1, -1)
        self.ri += 1
        self.ri %= 2
        return x
```
## Example Run
```
% python3.10 rolling_actions_dqn.py
n_episode :20, score : 42.5, n_buffer : 851, eps : 0.0%
n_episode :40, score : 37.2, n_buffer : 1596, eps : 0.0%
n_episode :60, score : 35.4, n_buffer : 2303, eps : 0.0%
n_episode :80, score : 30.1, n_buffer : 2904, eps : 0.0%
n_episode :100, score : 80.0, n_buffer : 4504, eps : 0.0%
n_episode :120, score : 206.6, n_buffer : 8635, eps : 0.0%
n_episode :140, score : 186.7, n_buffer : 12369, eps : 0.0%
n_episode :160, score : 129.6, n_buffer : 14961, eps : 0.0%
n_episode :180, score : 166.8, n_buffer : 18297, eps : 0.0%
n_episode :200, score : 133.1, n_buffer : 20959, eps : 0.0%
n_episode :220, score : 128.1, n_buffer : 23520, eps : 0.0%
n_episode :240, score : 239.7, n_buffer : 28313, eps : 0.0%
n_episode :260, score : 213.6, n_buffer : 32585, eps : 0.0%
n_episode :280, score : 248.4, n_buffer : 37554, eps : 0.0%
n_episode :300, score : 238.8, n_buffer : 42331, eps : 0.0%
n_episode :320, score : 171.0, n_buffer : 45751, eps : 0.0%
(infinite score achieved here)
```
