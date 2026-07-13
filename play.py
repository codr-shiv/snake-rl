import time
import pygame
import sys
from snake_env import SnakeEnv
from agent import Agent

def play():
    # Initialize the agent
    agent = Agent()
    agent.load('model.pth')
    
    # Try to initialize the environment with human rendering
    try:
        env = SnakeEnv(render_mode="human")
    except Exception as e:
        print("\n" + "=" * 50)
        print("ERROR: Failed to initialize display for rendering.")
        print("This usually happens when running in a headless environment (without X11/Wayland).")
        print("If you are inside an SSH or Docker session, you cannot view Pygame windows.")
        print(f"Details: {e}")
        print("=" * 50 + "\n")
        return

    print("Starting evaluation games...")
    print("Press CTRL+C in terminal or close the window to exit.")
    
    num_games = 10
    for game in range(1, num_games + 1):
        state, _ = env.reset()
        done = False
        score = 0
        
        while not done:
            # Handle Pygame events to allow closing the window gracefully
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("Exiting...")
                    env.close()
                    sys.exit()

            # Let agent choose action (no exploration)
            action = agent.get_action(state, train=False)
            
            # Step the environment
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            score = info['score']
            
            # Slight delay to make it watchable (speed reduced by half)
            time.sleep(0.1)
            
        print(f"Game {game} Finished! Score: {score}")
        time.sleep(1.0)
        
    env.close()

if __name__ == '__main__':
    play()
