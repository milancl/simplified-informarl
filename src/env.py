from .agent import Agent
from .gnn import grid_distance
import torch
import matplotlib.pyplot as plt
import itertools

import os.path as osp


class Environment:
    
    def __init__(self, config):
        self.config = config
        
        self.num_agents = config["num_agents"]
        self.grid_size = config["grid_size"]
        self.sensing_radius = config["sensing_radius"]
        self.max_obstacles = config["max_obstacles"]
        self.goal_reward = config["goal_reward"]
        self.collision_reward = config["collision_reward"]
        # self.max_distance = config.max_distance
        self.dim = config["dim"]
        self.agg_out_channels = config["agg_out_channels"]
        
        self.generate_action_map()
        
        self.init_entities()
        
        self.fig, self.ax = None, None


    def generate_action_map(self):
        # moves in each dimension: -1 (left/down), 0 (stay), 1 (right/up)
        moves = [-1, 0, 1]
        
        all_combinations = list(itertools.product(moves, repeat=self.dim))
        
        # create a dictionary
        self.action_to_move = {i: move for i, move in enumerate(all_combinations)}


    # initialize agents, goals, and obstacles positions
    def init_entities(self):
        def random_position():
            return torch.randint(0, self.grid_size, (self.dim,), dtype=torch.float)
        
        def already_selected(pos, all_pos):
            for p in all_pos:
                if torch.equal(pos, p):
                    return True
            return False
        
        selected_positions = []
        
        # initialize agents
        self.original_agents = []
        for _ in range(self.num_agents):
            while True:
                position = random_position()
                if not already_selected(position, selected_positions):
                    break
            self.original_agents.append(position)
            selected_positions.append(position)

        # initialize goals
        self.goals = []
        for _ in range(self.num_agents):
            while True:
                position = random_position()
                if not already_selected(position, selected_positions):
                    break
            self.goals.append(position)
            selected_positions.append(position)

        # initialize obstacles
        self.obstacles = []
        for _ in range(torch.randint(1, self.max_obstacles+1, (1,))):
            while True:
                position = random_position()
                if not already_selected(position, selected_positions):
                    break
            self.obstacles.append(position)
            selected_positions.append(position)
        
        # call reset to initialize agents
        self.reset()


    def reset(self, randomize=False):
        self.curr_step = 0
        if randomize:
            self.init_entities()
        else:
            self.agents = [
                Agent(position.detach().clone(), self.grid_size, dim=self.dim)  
                for position in self.original_agents 
            ]     


    def get_observations(self):
        # Returns a matrix N * d, where N is the number of agents and 
        # d is the observation vector dimension
        return torch.stack([
            torch.cat((agent.position, agent.velocity, goal - agent.position))
            for agent, goal in zip(self.agents, self.goals)
        ])


    def compute_rewards(self):
        # Returns:
        #   - the individual agents rewards
        #   - the joint reward
        
        collisions = 0
        
        rewards = torch.zeros(self.num_agents)
        dones = torch.zeros(self.num_agents)
        for i, (agent, goal) in enumerate(zip(self.agents, self.goals)):
            dist = grid_distance(goal - agent.position)
            if dist == 0:
                # add 5 reward for reaching the goal
                rewards[i] += self.goal_reward
                dones[i] = 1
            else:
                # negative reward based on goal distance
                # rewards[i] -= min(dist, self.max_distance)
                rewards[i] -= dist * 2 / self.grid_size

        # collision
        for i, agent_i in enumerate(self.agents):
            # other agents
            collided = False
            for agent_j in self.agents:
                if agent_i != agent_j and torch.equal(agent_i.position, agent_j.position):
                    collided = True
                    break
            if not collided:
                # obstacles
                for obs in self.obstacles:
                    if torch.equal(agent_i.position, obs):
                        collided = True
                        break
            
            if collided:
                rewards[i] += self.collision_reward
                collisions += 1
                
        
        return rewards, dones, collisions

    
    def get_inputs(self):
        observations = self.get_observations()
        actor_x_aggs, critic_X_aggs = self.get_aggregated_information()
        return observations, actor_x_aggs, critic_X_aggs


    def step(self, actions):
        assert actions.size(0) == self.num_agents
        
        # perform the move
        for agent, action in zip(self.agents, actions):
            agent.move(self.action_to_move[action.item()])

        return self.compute_rewards()

        
    def setup_plot(self):
        self.ax.set_xlim(-1, self.grid_size)
        self.ax.set_ylim(-1, self.grid_size)
        if self.dim == 3:
            self.ax.set_zlim(-1, self.grid_size)
        
        self.ax.set_aspect('equal')
        # self.ax.set_xticks(range(self.grid_size), minor=True)
        # self.ax.set_yticks(range(self.grid_size), minor=True)
        self.ax.set_xticks(range(0, self.grid_size, self.grid_size//10))
        self.ax.set_yticks(range(0, self.grid_size, self.grid_size//10))
        if self.dim == 3:
            # self.ax.set_zticks(range(self.grid_size), minor=True)
            self.ax.set_zticks(range(0, self.grid_size, self.grid_size//10))
            
            # Minor ticks will be where the grid lines are drawn
        self.ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
        self.ax.yaxis.set_minor_locator(plt.MultipleLocator(1))
        if self.dim == 3:
            self.ax.zaxis.set_minor_locator(plt.MultipleLocator(1))
        
        self.ax.grid(True, which='minor', linestyle='-', linewidth='0.5', color='gray')
        self.ax.grid(True, which='major', linestyle='-', linewidth='0.5', color='gray')



    def render(self, interactive=False, waiting_time=1.0, plotdir=None, title="InforMARL"):
        if self.fig is None:
            if self.dim == 2:
                self.fig, self.ax = plt.subplots()
            elif self.dim == 3:
                self.fig = plt.figure()
                self.ax = self.fig.add_subplot(111, projection='3d')
            else:
                raise NotImplementedError(f"Rendering with dimensions {self.dim} is not possible yet.")
        else:
            self.ax.cla()
        self.setup_plot()
        
        ms=10
        
        for agent, goal in zip(self.agents, self.goals):
            if self.dim == 2:
                goal_handle, = self.ax.plot(*goal, 'rs', markersize=ms, label="Goal")
                agent_handle, = self.ax.plot(*agent.position, 'g^', markersize=ms, label="Agent")
            elif self.dim == 3:
                goal_handle  = self.ax.scatter(*goal, 'rs', c='red', marker='o', s=100, label="Goal")
                agent_handle = self.ax.scatter(*agent.position, c='green', marker='o', s=50, label="Agent")
        
        for obs in self.obstacles:
            if self.dim == 2:
                obstacle_handle, = self.ax.plot(*obs, 'k8', markersize=ms, label="Obstacle")
            elif self.dim == 3:
                obstacle_handle = self.ax.scatter(*obs, 'k8', c='black', marker='o', s=100, label="Obstacle")
        
        if self.dim == 2: 
            self.ax.legend(handles=[agent_handle, goal_handle, obstacle_handle], loc='upper left', bbox_to_anchor=(1, 1))
        elif self.dim == 3:
            self.ax.legend([agent_handle, obstacle_handle, goal_handle], ['Agent', 'Obstacle', 'Goal'])
        self.fig.suptitle(title, fontsize=15)
        
        if interactive:
            plt.pause(waiting_time)  # Pause a bit so that updates are visible
            plt.draw()
        else:
            self.fig.savefig(osp.join(plotdir, f"{self.curr_step:03d}.png"))
            self.curr_step += 1
            
        
