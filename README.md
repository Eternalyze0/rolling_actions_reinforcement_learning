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
