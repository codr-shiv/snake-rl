# Reinforcement Learning Snake Agent

This project applies reinforcement learning to train a DQN agent to play the snake game on a 20x20 wrapping grid. The grid features randomly generated food, poison, and straight wall cells. The reward policy gives -200 for eating itself or hitting a wall, +50 for eating food, and -20 for consuming poison.

To improve performance and prevent looping behaviors, the implementation includes:
- A potential-based reward shaping system which rewards moving closer to the food at each step.
- A 3-directional Lidar sensor system to give the agent spatial depth awareness.
- A BFS path check to detect and avoid self-trapping loops.
- Tail-popping physics to allow the snake to safely follow its own tail.

The stack used is PyTorch and Gymnasium for the reinforcement learning components, and Pygame for the visual interface.

The agent is trained over 1000 episodes using epsilon-greedy exploration. The results are:
- Max score: 93
- Final running average: 35.6
- Peak exploitation average: 46.9

## How to Run

1. Set up a virtual environment and activate it:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Train the model:
```bash
python train.py
```

4. Watch the trained model play:
```bash
python play.py
```
