# Reinforcement Learning Snake Agent

This project applies reinforcement learning to train a DQN agent to play the snake game on a 20x20 wrapping grid. The grid features randomly generated food, poison, and straight wall cells. The reward policy gives -200 for eating itself or hitting a wall, +50 for eating food, and -20 for consuming poison.

To improve performance and prevent looping behaviors, the implementation includes:
- A potential-based reward shaping system which rewards moving closer to the food at each step.
- A 3-directional Lidar sensor system to give the agent spatial depth awareness.
- A BFS path check to detect and avoid self-trapping loops.
- Tail-popping physics to allow the snake to safely follow its own tail.

The stack used is PyTorch and Gymnasium for the reinforcement learning components, and Pygame for the visual interface.

Features

Lidar Strategy
The snake looks in three directions relative to its head: straight, right, and left. Instead of just seeing immediate neighbors, it projects raycasts to measure wrapping distance to the nearest obstacle (body, wall, or poison), returning a continuous proximity value (1.0 / distance).

BFS Survival Strategy
To prevent the snake from trapping itself in dead-ends, a breadth-first search is run from the head on each step. If the reachable open space is smaller than the snake's current length, the agent knows it is entering a trap and is penalized, encouraging it to select wider paths.

Potential-Based Reward Shaping
Instead of hardcoded progress rewards, we use potential-based reward shaping where potential is defined as the negative wrapping distance to the food. The shaping reward added to the step is calculated as (gamma * -new_distance) - (-old_distance) with a gamma of 0.97. This guides the snake to the food without changing the optimal policy or introducing loops. A step penalty of -0.2 is also applied.

Neural Network to predict Q-values
- Input: 12 values (3 Lidar proximity readings, 1 BFS path existence check, 4 heading directions, 4 relative food directions)
- Hidden Layers: 256 ReLU units -> 256 ReLU units
- Output Layer: 3 units (Q-values for straight, turn right, and turn left actions)

Training Process
The agent trains over 1000 episodes using epsilon-greedy exploration. Epsilon decays from 1.0 down to 0.01 at a rate of 0.994 per episode.

The results are:
- Max score: 93
- Final running average: 35.6
- Peak exploitation average: 46.9

![Training Progress](training_progress.png)

How to Run

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
