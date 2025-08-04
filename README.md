## AtariAI_DQN
This project is created based on [Playing Atari with deep RL](https://github.com/shubhlohiya/playing-atari-with-deep-RL), as a reproduction of [Playing Atari with Deep Reiforcement Learing](https://arxiv.org/abs/1312.5602)

---
### Repository Structure 
```
DQN
├── model.py
├── models
│   ├── Boxing
│   │   ├── # saved model weights checkpoints for Boxing
│   ├── Breakout
│   │   ├── # saved model weights checkpoints for Breakout
│   └── Pong
│       ├── # saved model weights checkpoints for Pong
├── params.py
├── player.py
├── __pycache__
│   ├── # complied results
├── replay.py
├── trainer.py
├── videos
│   ├── Boxing
│   │   ├── # videos of agent playing Boxing games
│   ├── Breakout
│   │   ├── # videos of agent playing Breakout games
│   └── Pong
│       ├── # videos of agent playing Pong games
└── wrappers.py
```
---
### Usage
If you want to work on this repository or train your own models, you can simply change the `ModelTrainer(env_name=['your_atari_name'])` in `trainer.py` and get the result of agent playing game with `ModelPlayer` in `player.py`