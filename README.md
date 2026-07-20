# Reinforcement Learning Snake Agent

This project implements a Deep Q-Network (DQN) RL agent to play the snake game on a custom 20x20 wrapping grid environment. Poison, food, and continuous walls are generated randomly. The base reward function gives -200 for eating itself or hitting a wall, +50 for food, and -20 for poison.

To improve policy performance and prevent reward hacking, the implementation includes:

1. **Potential-Based Reward Shaping (PBRS)**: Rewards moving closer to the food at each step using the negative wrapping distance: $F(s, a, s') = \gamma \Phi(s') - \Phi(s)$ where $\Phi(s) = -\text{distance}$ and $\gamma = 0.97$. This guides the agent without altering the optimal policy or introducing positive reward cycles.
2. **Step Penalty & Starvation Timer**: Applies a `-0.2` step penalty and truncates episodes with a `-200` penalty if the snake takes $> 100 \times \text{body\_length}$ steps without eating food (resetting upon eating food), effectively preventing infinite looping.
3. **3-Directional Lidar**: Provides vision by projecting raycasts in straight, left, and right directions to measure proximity ($1.0 / \text{distance}$) to walls, body segments, or poison.
4. **BFS Path Check**: Performs a breadth-first search from the head on each step to check for topological connectivity (verifying reachable space $\ge \text{body\_length}$), preventing the snake from trapping itself in dead-ends.
5. **Tail-Popping Physics**: Removes the tail segment before verifying collisions so the snake can safely follow its own body.
6. **Target Network Soft Updates**: Uses Polyak averaging ($\tau = 0.01$) to update the target Q-network, ensuring smooth Q-learning stability.

The stack used is PyTorch and Gymnasium for RL, and Pygame for GUI rendering.

## Neural Network Architecture

- **Input Layer**: 12 feature values (3 Lidar proximity readings, 1 BFS survival path status, 4 current direction flags, and 4 relative food direction flags).
- **Hidden Layers**: Two fully connected layers with 256 ReLU units each.
- **Output Layer**: 3 Q-value predictions corresponding to actions: `[Straight, Turn Right, Turn Left]`.

## Training Process and Results

The policy was trained over 1000 episodes using epsilon-greedy exploration ($\epsilon$ decaying from 1.0 to 0.01).

### Performance Metrics:
- **Max score**: 83 (food items eaten in a single game)
- **Final running average score (last 10 games)**: 44.70
- **Peak running average score**: 51.40

![Training Progress](training_progress.png)

## Loss Function & Training Stability Analysis

The training loss is measured using Mean Squared Error (MSE) between the target Q-values (computed via Bellman equation using target network) and predicted Q-values.

![Training Loss Curve](loss_curve.png)

### Key Insights & Justification:
- **Convergence and Stability**: The loss starts high (~477.8) during initial exploration as the network encounters large penalty signals (-200). It quickly drops and stabilizes in the **35–60 MSE range** (finishing with a 10-episode mean loss of **51.10** and final loss of **38.95**), indicating smooth policy convergence without loss divergence.
- **Overestimation Control**: Using soft updates ($\tau = 0.01$) on the target network prevents Q-value overestimation spikes, maintaining steady gradients throughout the 1000 episodes.
- **Absence of Reward Hacking**: The combination of Potential-Based Reward Shaping, a `-0.2` step penalty, and the starvation timeout ($100 \times \text{body\_length}$) prevents the agent from engaging in loop farming. As seen in evaluation play, the agent directly navigates to food while keeping topological escape routes open.

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
