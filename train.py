import matplotlib
matplotlib.use('Agg')  # Headless backend to prevent display errors
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from snake_env import SnakeEnv
from agent import Agent

def plot(scores, mean_scores):
    plt.figure(figsize=(10, 6))
    plt.title('DQN Snake Training Progress (Walls & Poison)', fontsize=14, fontweight='bold')
    plt.xlabel('Games / Episodes', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    
    # Plotting scores and moving average
    plt.plot(scores, color='#52c759', alpha=0.3, label='Game Score')
    plt.plot(mean_scores, color='#ff453a', linewidth=2, label='Running Average (10 games)')
    
    # Highlight final values
    if len(scores) > 0:
        plt.text(len(scores) - 1, scores[-1], f"Last: {scores[-1]}", fontsize=9, fontweight='bold', color='#2c3e50')
    if len(mean_scores) > 0:
        plt.text(len(mean_scores) - 1, mean_scores[-1], f"Mean: {mean_scores[-1]:.2f}", fontsize=9, fontweight='bold', color='#e74c3c')

    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('training_progress.png', dpi=150)
    plt.close()

def plot_loss(losses, mean_losses):
    plt.figure(figsize=(10, 6))
    plt.title('DQN Snake Training Loss Curve', fontsize=14, fontweight='bold')
    plt.xlabel('Games / Episodes', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    
    # Plotting losses and moving average
    plt.plot(losses, color='#3498db', alpha=0.3, label='Episode Loss')
    plt.plot(mean_losses, color='#e67e22', linewidth=2, label='Running Average (10 games)')
    
    # Highlight final values
    if len(losses) > 0:
        plt.text(len(losses) - 1, losses[-1], f"Last: {losses[-1]:.2f}", fontsize=9, fontweight='bold', color='#2c3e50')
    if len(mean_losses) > 0:
        plt.text(len(mean_losses) - 1, mean_losses[-1], f"Mean: {mean_losses[-1]:.2f}", fontsize=9, fontweight='bold', color='#d35400')

    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('loss_curve.png', dpi=150)
    plt.close()

def train():
    plot_scores = []
    plot_mean_scores = []
    plot_losses = []
    plot_mean_losses = []
    total_score = 0
    record = 0
    
    env = SnakeEnv()
    agent = Agent()
    
    print("Starting training with walls, poison, and reward shaping...")
    print("Target: Teach the DQN agent to navigate obstacles and eat food.")
    print("-" * 60)
    
    # Run for 1000 games
    num_episodes = 1000
    
    for episode in range(1, num_episodes + 1):
        # Reset environment
        state, _ = env.reset()
        done = False
        
        while not done:
            # Get old state
            state_old = state
            
            # Get action based on state
            action = agent.get_action(state_old, train=True)
            
            # Perform action and get new state
            state_new, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Train short memory (single step)
            agent.train_short_memory(state_old, action, reward, state_new, done)
            
            # Remember transition
            agent.remember(state_old, action, reward, state_new, done)
            
            state = state_new
            
        # Game finished
        score = info['score']
        agent.n_games += 1
        loss = agent.train_long_memory()
        agent.decay_epsilon()
        
        if score > record:
            record = score
            agent.save()  # Save the best model
            
        plot_scores.append(score)
        total_score += score
        
        # Calculate moving average of last 10 games
        mean_score = np.mean(plot_scores[-10:])
        plot_mean_scores.append(mean_score)
        
        # Track loss
        plot_losses.append(loss)
        mean_loss = np.mean(plot_losses[-10:])
        plot_mean_losses.append(mean_loss)
        
        # Print progress every game
        if episode % 10 == 0 or episode == 1:
            print(f"Game: {episode:3d} | Score: {score:3d} | Record: {record:3d} | Epsilon: {agent.epsilon:.4f} | Loss: {loss:.4f} | Running Avg: {mean_score:.2f}")
            plot(plot_scores, plot_mean_scores)
            plot_loss(plot_losses, plot_mean_losses)
            
    env.close()
    
    # Save training history data to json
    history = {
        "scores": plot_scores,
        "mean_scores": plot_mean_scores,
        "losses": plot_losses,
        "mean_losses": plot_mean_losses
    }
    with open("training_history.json", "w") as f:
        json.dump(history, f)
        
    print("-" * 60)
    print("Training finished!")
    print(f"Best score achieved: {record}")
    print(f"Final running average score: {plot_mean_scores[-1]:.2f}")
    print("Model saved to 'model/model.pth'")
    print("Progress plot saved to 'training_progress.png'")
    print("Loss curve saved to 'loss_curve.png'")
    print("History saved to 'training_history.json'")

if __name__ == '__main__':
    train()
