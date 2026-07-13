import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random

class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 15}

    def __init__(self, grid_size=20, cell_size=20, num_walls=3, num_poisons=5, min_wall_len=2, max_wall_len=6, render_mode=None):
        super(SnakeEnv, self).__init__()
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.num_walls = num_walls
        self.num_poisons = num_poisons
        self.min_wall_len = min_wall_len
        self.max_wall_len = max_wall_len
        self.render_mode = render_mode

        # 3 Actions: 0: Straight, 1: Turn Right, 2: Turn Left
        self.action_space = spaces.Discrete(3)

        # 12-value observation representation
        self.observation_space = spaces.Box(low=0, high=1, shape=(12,), dtype=np.float32)

        # Pygame parameters
        self.window_size = self.grid_size * self.cell_size
        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        center_x = self.grid_size // 2
        center_y = self.grid_size // 2

        # Initialize snake: start in center heading Right
        self.body = [
            [center_x, center_y],
            [center_x - 1, center_y],
            [center_x - 2, center_y]
        ]
        self.direction = [1, 0]  # [dx, dy] facing right

        self.score = 0
        self.frame_iteration = 0

        # Spawn random continuous straight walls (no intersecting corners)
        self.walls = []
        actual_num_walls = random.randint(2, 4)
        for _ in range(actual_num_walls):
            for attempt in range(100):
                x = random.randint(1, self.grid_size - 2)
                y = random.randint(1, self.grid_size - 2)
                
                # Direction: 0 for Horizontal, 1 for Vertical
                dir_type = random.choice([0, 1])
                length = random.randint(self.min_wall_len, self.max_wall_len)
                
                line_pts = []
                valid = True
                for step in range(length):
                    px = x + (step if dir_type == 0 else 0)
                    py = y + (0 if dir_type == 0 else step)
                    
                    if px < 0 or px >= self.grid_size or py < 0 or py >= self.grid_size:
                        valid = False
                        break
                        
                    if abs(px - center_x) <= 3 and abs(py - center_y) <= 3:
                        valid = False
                        break
                        
                    if [px, py] in self.walls:
                        valid = False
                        break
                        
                    # Prevent diagonal or adjacent corner touching
                    for wx, wy in self.walls:
                        if abs(wx - px) <= 1 and abs(wy - py) <= 1:
                            valid = False
                            break
                    if not valid:
                        break
                        
                    line_pts.append([px, py])
                
                if valid and len(line_pts) == length:
                    self.walls.extend(line_pts)
                    break

        # Spawn food
        self.food = None
        self._place_food()

        # Spawn poisons
        self.poisons = []
        self._place_poisons()

        observation = self._get_obs()
        info = {}

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def _place_food(self):
        while True:
            x = random.randint(0, self.grid_size - 1)
            y = random.randint(0, self.grid_size - 1)
            self.food = [x, y]
            if self.food not in self.body and self.food not in self.walls:
                if not hasattr(self, 'poisons') or self.food not in self.poisons:
                    break

    def _place_poisons(self):
        self.poisons = []
        while len(self.poisons) < self.num_poisons:
            x = random.randint(0, self.grid_size - 1)
            y = random.randint(0, self.grid_size - 1)
            pt = [x, y]
            if (pt not in self.body and 
                pt not in self.walls and 
                pt != self.food and 
                pt not in self.poisons):
                self.poisons.append(pt)

    def _get_wrapping_distance(self, pt1, pt2):
        dx = abs(pt1[0] - pt2[0])
        dy = abs(pt1[1] - pt2[1])
        dx = min(dx, self.grid_size - dx)
        dy = min(dy, self.grid_size - dy)
        return dx + dy

    def _get_raycast_proximity(self, start, direction):
        curr = list(start)
        max_dist = self.grid_size
        obstacles = set(tuple(p) for p in self.walls)
        obstacles.update(tuple(p) for p in self.poisons)
        obstacles.update(tuple(p) for p in self.body[:-1])
        
        for dist in range(1, max_dist + 1):
            curr[0] = (curr[0] + direction[0]) % self.grid_size
            curr[1] = (curr[1] + direction[1]) % self.grid_size
            if tuple(curr) in obstacles:
                return 1.0 / dist
        return 0.0

    def _check_survival_path(self, start_pos):
        # BFS to find reachable free space
        queue = [tuple(start_pos)]
        visited = {tuple(start_pos)}
        reachable_count = 0
        
        obstacles = set(tuple(p) for p in self.walls)
        obstacles.update(tuple(p) for p in self.poisons)
        obstacles.update(tuple(p) for p in self.body[:-1])
        
        while queue:
            curr = queue.pop(0)
            reachable_count += 1
            if reachable_count >= len(self.body):
                return 1.0
                
            for dx, dy in [[1, 0], [-1, 0], [0, 1], [0, -1]]:
                neighbor = ((curr[0] + dx) % self.grid_size, (curr[1] + dy) % self.grid_size)
                if neighbor not in visited and neighbor not in obstacles:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return 0.0

    def _get_obs(self):
        head = self.body[0]
        
        # Determine current directions
        dir_l = self.direction == [-1, 0]
        dir_r = self.direction == [1, 0]
        dir_u = self.direction == [0, -1]
        dir_d = self.direction == [0, 1]

        # Get relative direction vectors
        clockwise_dirs = [[1, 0], [0, 1], [-1, 0], [0, -1]]
        idx = clockwise_dirs.index(self.direction)
        
        dir_straight = self.direction
        dir_right = clockwise_dirs[(idx + 1) % 4]
        dir_left = clockwise_dirs[(idx - 1) % 4]

        # Raycasts for proximity detection (1.0 / distance)
        prox_straight = self._get_raycast_proximity(head, dir_straight)
        prox_right = self._get_raycast_proximity(head, dir_right)
        prox_left = self._get_raycast_proximity(head, dir_left)

        # BFS survival path connectivity check
        survival_path = self._check_survival_path(head)

        # Shortest wrapping food vector
        dx = self.food[0] - head[0]
        dy = self.food[1] - head[1]
        
        if abs(dx) > self.grid_size / 2:
            dx = -int(np.sign(dx)) * (self.grid_size - abs(dx))
        if abs(dy) > self.grid_size / 2:
            dy = -int(np.sign(dy)) * (self.grid_size - abs(dy))

        food_straight = 0
        food_right = 0
        food_left = 0
        food_back = 0

        if dir_r:
            food_straight = int(dx > 0)
            food_back = int(dx < 0)
            food_right = int(dy > 0)
            food_left = int(dy < 0)
        elif dir_l:
            food_straight = int(dx < 0)
            food_back = int(dx > 0)
            food_right = int(dy < 0)
            food_left = int(dy > 0)
        elif dir_u:
            food_straight = int(dy < 0)
            food_back = int(dy > 0)
            food_right = int(dx > 0)
            food_left = int(dx < 0)
        elif dir_d:
            food_straight = int(dy > 0)
            food_back = int(dy < 0)
            food_right = int(dx < 0)
            food_left = int(dx > 0)

        state = [
            prox_straight,
            prox_right,
            prox_left,
            float(survival_path),
            
            float(dir_l),
            float(dir_r),
            float(dir_u),
            float(dir_d),
            
            float(food_straight),
            float(food_right),
            float(food_left),
            float(food_back)
        ]

        return np.array(state, dtype=np.float32)

    def _is_collision(self, pt=None):
        if pt is None:
            pt = self.body[0]
            body_to_check = self.body[1:]
        else:
            body_to_check = self.body[:-1]
        
        # Check internal wall collision
        if pt in self.walls:
            return True
            
        # Check body collision
        if pt in body_to_check:
            return True
            
        return False

    def _is_danger(self, pt):
        # Danger includes collisions and poison cells
        if self._is_collision(pt):
            return 1
        if pt in self.poisons:
            return 1
        return 0

    def step(self, action):
        self.frame_iteration += 1

        # Determine new direction
        clockwise_dirs = [[1, 0], [0, 1], [-1, 0], [0, -1]]
        idx = clockwise_dirs.index(self.direction)

        if action == 0:
            new_dir = self.direction
        elif action == 1:
            new_dir = clockwise_dirs[(idx + 1) % 4]
        else:
            new_dir = clockwise_dirs[(idx - 1) % 4]

        self.direction = new_dir

        # Move head with modulo wrapping for continuous/wrapping grid
        head = self.body[0]
        new_head = [
            (head[0] + self.direction[0]) % self.grid_size,
            (head[1] + self.direction[1]) % self.grid_size
        ]
        
        # Calculate old distance to food using wrapping metric
        old_dist = self._get_wrapping_distance(head, self.food)
        new_dist = self._get_wrapping_distance(new_head, self.food)

        # Insert new head
        self.body.insert(0, new_head)

        reward = 0.0
        terminated = False
        truncated = False

        # Determine if head is on food or poison
        is_food = (new_head == self.food)
        is_poison = (new_head in self.poisons)

        # Update body segments (pop tail) before collision checks if NOT eating food
        if not is_food:
            self.body.pop()
            if is_poison:
                if len(self.body) > 1:
                    self.body.pop()

        # 1. Collision check (game over)
        if self._is_collision():
            reward = -200.0  # High penalty for collision!
            terminated = True
            self.body.pop(0)  # Remove the collided head for visualization safety
            return self._get_obs(), reward, terminated, truncated, {"score": self.score}

        # 2. Food check
        if is_food:
            self.score += 1
            reward = 50.0
            self._place_food()
            self._place_poisons()
            self.frame_iteration = 0  # Reset starvation timer
            
        # 3. Poison check
        elif is_poison:
            reward = -20.0
            # Check if snake is too small after poison shrink
            if len(self.body) < 2:
                terminated = True
                reward = -200.0  # Game over penalty
            else:
                self._place_poisons()
                
        # 4. Normal Step
        else:
            # Potential-based reward shaping using gamma = 0.97
            gamma = 0.97
            shaping_reward = (gamma * -new_dist) - (-old_dist)
            reward = shaping_reward - 0.2

        # Truncation check
        if self.frame_iteration > 100 * len(self.body):
            truncated = True
            reward = -200.0

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, terminated, truncated, {"score": self.score}

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Snake RL Environment (Walls + Poison)")
            self.clock = pygame.time.Clock()

        if self.window is None and self.render_mode == "rgb_array":
            self.window = pygame.Surface((self.window_size, self.window_size))

        # Premium UI Colors
        COLOR_BG = (21, 21, 23)       # Dark charcoal
        COLOR_GRID = (32, 32, 35)     # Subtle grid lines
        COLOR_WALL = (120, 120, 128)  # Steel gray
        COLOR_FOOD = (255, 69, 58)    # Vibrant coral red
        COLOR_POISON = (191, 90, 242) # Premium electric violet/purple
        
        self.window.fill(COLOR_BG)

        # Draw grid
        for i in range(self.grid_size):
            pygame.draw.line(self.window, COLOR_GRID, (i * self.cell_size, 0), (i * self.cell_size, self.window_size))
            pygame.draw.line(self.window, COLOR_GRID, (0, i * self.cell_size), (self.window_size, i * self.cell_size))

        # Draw Walls
        for wall in self.walls:
            rect = pygame.Rect(
                wall[0] * self.cell_size + 1,
                wall[1] * self.cell_size + 1,
                self.cell_size - 2,
                self.cell_size - 2
            )
            pygame.draw.rect(self.window, COLOR_WALL, rect, border_radius=3)

        # Draw Poisons
        for poison in self.poisons:
            # Draw poison as a diamond or custom shape
            points = [
                (poison[0] * self.cell_size + self.cell_size / 2, poison[1] * self.cell_size + 1),
                (poison[0] * self.cell_size + self.cell_size - 1, poison[1] * self.cell_size + self.cell_size / 2),
                (poison[0] * self.cell_size + self.cell_size / 2, poison[1] * self.cell_size + self.cell_size - 1),
                (poison[0] * self.cell_size + 1, poison[1] * self.cell_size + self.cell_size / 2)
            ]
            pygame.draw.polygon(self.window, COLOR_POISON, points)

        # Draw Food
        food_center = (
            int(self.food[0] * self.cell_size + self.cell_size / 2),
            int(self.food[1] * self.cell_size + self.cell_size / 2)
        )
        pygame.draw.circle(self.window, COLOR_FOOD, food_center, int(self.cell_size / 2 - 2))

        # Draw Snake (Gradient from neon cyan/blue to deep blue)
        for idx, segment in enumerate(self.body):
            if idx == 0:
                color = (0, 199, 190)   # Bright Cyan head
            else:
                factor = max(0.3, 1.0 - (idx / len(self.body)) * 0.7)
                color = (0, int(122 * factor), int(255 * factor)) # Blue tail gradient

            rect = pygame.Rect(
                segment[0] * self.cell_size + 1,
                segment[1] * self.cell_size + 1,
                self.cell_size - 2,
                self.cell_size - 2
            )
            pygame.draw.rect(self.window, color, rect, border_radius=4)

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
