# Reinforcement Learning Snake Agent

This project is a simple reinforcement learning agent that learns to play Snake in a wrapping grid with obstacles. The goal of the agent is to eat food and avoid crashing into obstacles or itself.

## Tech Stack

The code is in Python using PyTorch for the deep learning model, Gymnasium for the reinforcement learning environment wrapper, and Pygame to render the game screen.

## How the Game Works

The game environment has a few specific rules to help the agent learn:

Wrapping Grid
If the snake head goes off any of the grid boundaries, it wraps around to the opposite side. There are no wall collisions at the grid borders.

Tail Evasion
To let the snake follow its tail when it gets long, we update and pop the tail segment before verifying collisions. This prevents the snake from crashing into its own tail when moving into the space it just vacated.

Lidar Sensors
The snake looks in three directions relative to its head: straight, right, and left. Instead of just seeing what is next to it, it projects raycasts to measure the distance to the nearest danger, giving it depth awareness.

Survival Check
To prevent the snake from trapping itself in dead-ends, we run a quick breadth-first search from the head on each step. If the reachable open space is smaller than the snake's current length, the agent knows it is entering a trap.

Random Walls and Poisons
On each reset, a random number of straight wall segments are generated with random lengths, placed in a way that prevents them from forming corners. Five poison cells also spawn, which shrink the snake if consumed.

## Reinforcement Learning Setup

The agent uses a Deep Q-Network with a target network and soft updates. The model takes a 12-value observation vector containing the Lidar readings, the survival check status, the direction of the snake, and the relative direction of the food.

Instead of using hardcoded step rewards, we use potential-based reward shaping. The potential is defined as the negative wrapping distance to the food. The shaping reward added to the step is calculated as:
(gamma * -new_distance) - (-old_distance)
where gamma is 0.97. This guides the snake to the food without changing the optimal policy or introducing loops. A step penalty of -0.2 is also applied.

## Results

After training for 1000 episodes, the agent achieves a max score of 93, final running average 35.6, and peak exploitation average 46.9.

## How to Run

1. Create a virtual environment and activate it:
python3 -m venv venv
source venv/bin/activate

2. Install the requirements:
pip install -r requirements.txt

3. Train the model:
python train.py

4. Watch the trained model play:
python play.py
