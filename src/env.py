import shutil
from .agent import Agent
from .gnn import construct_graph, UniMPGNN, grid_distance
import torch
import torch.nn.functional as F
from .net import Actor, Critic
import matplotlib.pyplot as plt
import itertools

from torch_geometric.loader import DataLoader
from torch.distributions import Categorical

import os.path as osp
import os

import cv2



class Environment:
    
    def __init__(self, num_agents, grid_size, sensing_radius, 
                 agg_out_channels=32, dim=2, max_obstacles=4, 
                 goal_reward=5, collision_reward=-5, gamma=0.8,
                 max_distance=5):
        self.num_agents = num_agents
        self.grid_size = grid_size
        self.sensing_radius = sensing_radius
        
        self.max_obstacles = max_obstacles
        
        self.goal_reward = goal_reward
        self.collision_reward = collision_reward
        self.gamma = gamma
        self.max_distance = max_distance
        
        self.dim = dim
        self.generate_action_map()
        
        self.actor = Actor(
            agg_out_channels + 3 * self.dim, # agg dimension + (pos, vel, rel_goal) * self.dim
            len(self.action_to_move)
        )
        self.actor_gnn = UniMPGNN(
            3 * self.dim + 3,  # (rel_pos, vel, rel_goal) * self.dim + entity embedding
            agg_out_channels,
            aggregate=False
        )
        
        self.critic = Critic(
            agg_out_channels
        )
        self.critic_gnn = UniMPGNN(
            3 * self.dim + 3,  # (rel_pos, vel, rel_goal) * self.dim + entity embedding
            agg_out_channels,
            aggregate=True
        )
        
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()) + 
            list(self.actor_gnn.parameters()) + list(self.critic_gnn.parameters()),
            lr=0.0001
        )
        
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


    def reset(self, randomized=False):
        self.curr_step = 0
        if randomized:
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


    def get_aggregated_information(self):
        # Returns a matrix N * c, where N is the number of agents and 
        # c is the aggregated vector dimension
        
        # X = []
        # for agent in self.agents:
        #     data = construct_graph(
        #         agent, 
        #         self.agents, 
        #         self.goals, 
        #         self.obstacles, 
        #         self.sensing_radius
        #     )
        #     X.append(
        #         self.information_aggregation(data) # GNN 
        #     )
        # return X
        
        loader = DataLoader([
            construct_graph(
                i, self.agents, self.goals, self.obstacles, self.sensing_radius, dim=self.dim
            )
            for i in range(len(self.agents))
        ], batch_size=len(self.agents))
        batch = next(iter(loader))
        
        actor_x_aggs = self.actor_gnn(batch)
        critic_X_aggs = self.critic_gnn(batch)
        
        return actor_x_aggs, critic_X_aggs

    def compute_rewards(self):
        # Returns:
        #   - the individual agents rewards
        #   - the joint reward
        
        collisions = 0
        
        rewards = torch.zeros(len(self.agents))
        dones = torch.zeros(len(self.agents))
        for i, (agent, goal) in enumerate(zip(self.agents, self.goals)):
            dist = grid_distance(goal - agent.position)
            if dist == 0:
                # add 5 reward for reaching the goal
                rewards[i] += self.goal_reward
                dones[i] = 1
            else:
                # negative reward based on goal distance
                # rewards[i] -= min(dist, self.max_distance)
                rewards[i] -= dist

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


    def step(self):
        observations = self.get_observations()
        actor_x_aggs, critic_X_aggs = self.get_aggregated_information()
        
        # N * [obs(i), x_agg(i)]
        actor_inputs = torch.cat((observations, actor_x_aggs), dim=1)
        critic_inputs = critic_X_aggs
        
        # get action probabilities and sample one action per agent
        action_logits = self.actor(actor_inputs)
        # action_probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(logits=action_logits)
        actions = dist.sample()
        assert actions.size(0) == len(self.agents)

        # get state values 
        state_values = self.critic(critic_inputs)
        assert state_values.size(0) == len(self.agents)
        
        # perform the move
        for agent, action in zip(self.agents, actions):
            agent.move(self.action_to_move[action.item()])

        rewards, dones, collisions = self.compute_rewards()
        
        log_probs = dist.log_prob(actions)

        return actions, log_probs, state_values, rewards, dones, collisions
    
    
    def train(self, num_episodes, num_steps, run_name, 
                randomize_every=100, eval_every=1000, 
                save_models=True, save_every=1000, randomize=False):
        for episode in range(num_episodes):
            # reset environment some times
            if randomize and (episode+1) % randomize_every == 0:
                self.reset(randomized=True)  
            else:
                self.reset()  
            
            # store episode data
            log_probs = []
            values = []
            rewards = []
            masks = []

            for step in range(num_steps):
                actions, log_prob, state_values, reward, dones, _ = self.step() 
                
                log_probs.append(log_prob)
                values.append(state_values)
                rewards.append(reward.unsqueeze(1))
                masks.append(1 - dones.unsqueeze(1))
            
            # convert to tensors
            log_probs = torch.cat(log_probs)
            values = torch.cat(values) 
            rewards = torch.cat(rewards)
            masks = torch.cat(masks)

            # compute returns (start from the last to compute the correct gamma weighting)
            returns = []
            R = 0
            for step in reversed(range(len(rewards))):
                R = rewards[step] + self.gamma * R * masks[step]
                returns.insert(0, R)

            # Advantages are compute as the expected critic reward, and the actual 
            # return computed in the environment. If an advantage is positive, it reduces 
            # the loss, meaning that a good action was taken. If negative, they increase the 
            # loss, meaning a bad action was chosen
            returns = torch.cat(returns)
            advantages = returns - values

            # Calculate losses
            actor_loss = -(log_probs * advantages.detach()).mean()
            # values and returns should have the same shape
            critic_loss = F.mse_loss(values.squeeze(), returns.detach())
            loss = actor_loss + critic_loss

            # Perform backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Logging and saving models
            if (episode+1) % eval_every == 0:
                print(f"Episode: {episode+1}, Loss: {loss.item()}")
                self.eval(
                    num_steps=num_steps,
                    run_name=run_name,
                    randomize=False, 
                    render=False, 
                    load=False
                )
            
            if save_models and (episode+1) % save_every == 0:
                self.save_models(run_name)

    
    def eval(self, num_steps, run_name, randomize=False, render=False, load=False):
        if load:
            self.load_models(run_name)

        if render:
            # save plots images in a tmp dir
            plotdir = "tmp"
            if not os.path.exists(plotdir):
                os.mkdir(plotdir)

        # reset the environment with randomized settings (new agents positions)
        self.reset(randomized=randomize)
        episode_collisions = 0
        total_rewards = 0
        
        for s in range(num_steps):
            _, _, _, rewards, dones, collisions = self.step()
            
            total_rewards += rewards.sum().item()
            episode_collisions += collisions
            
            if render:
                self.render(plotdir=plotdir, title=f"Step: {s+1}/{num_steps}\n" + 
                                f"Total collisions: {episode_collisions} -- Done: {int(dones.sum())}/{self.num_agents} agents")
        
        print(f"Episode reward: {total_rewards} -- Collisions: {episode_collisions} -- Done: {dones.sum()}/{self.num_agents} agents")
        
        if render:
            # If rendering, perhaps convert saved images to a video or similar
            self.save_video(run_name, plotdir, fps=6)  # Make sure you have a function `save_video`
            # Clean up after rendering
            shutil.rmtree(plotdir)
            
    def load_models(self, run_name):
        model_dir = osp.join("runs", run_name)
        assert osp.exists(model_dir)
        
        self.actor.load_state_dict(torch.load(osp.join(model_dir, "actor.pth")))
        self.critic.load_state_dict(torch.load(osp.join(model_dir, "critic.pth")))
        self.actor_gnn.load_state_dict(torch.load(osp.join(model_dir, "actor_gnn.pth")))
        self.critic_gnn.load_state_dict(torch.load(osp.join(model_dir, "critic_gnn.pth")))
        
        self.actor.eval()
        self.critic.eval()
        self.actor_gnn.eval()
        self.critic_gnn.eval()


    def save_models(self, run_name):
        model_dir = osp.join("runs", run_name)
        if not osp.isdir(model_dir):
            os.mkdir(model_dir)
        torch.save(self.actor.state_dict(), osp.join(model_dir, "actor.pth"))
        torch.save(self.critic.state_dict(), osp.join(model_dir, "critic.pth"))
        torch.save(self.actor_gnn.state_dict(), osp.join(model_dir, "actor_gnn.pth"))
        torch.save(self.actor_gnn.state_dict(), osp.join(model_dir, "critic_gnn.pth"))


    def save_video(self, run_name, plotdir, fps=1):
        run_dir = osp.join("runs", run_name)
        image_folder = plotdir
        video_name = osp.join(run_dir, "video.mp4")

        images = sorted([img for img in os.listdir(image_folder) if img.endswith(".png")])
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, _ = frame.shape

        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        video = cv2.VideoWriter(video_name, fourcc, fps, (width,height))

        for image in images:
            video.write(cv2.imread(os.path.join(image_folder, image)))

        video.release()
        cv2.destroyAllWindows()
        

    def setup_plot(self):
        self.ax.set_xlim(-1, self.grid_size)
        self.ax.set_ylim(-1, self.grid_size)
        if self.dim == 3:
            self.ax.set_zlim(-1, self.grid_size)
        
        self.ax.set_aspect('equal')
        self.ax.set_xticks(range(self.grid_size), minor=True)
        self.ax.set_yticks(range(self.grid_size), minor=True)
        self.ax.set_xticks(range(0, self.grid_size, 10))
        self.ax.set_yticks(range(0, self.grid_size, 10))
        if self.dim == 3:
            self.ax.set_zticks(range(self.grid_size), minor=True)
            self.ax.set_zticks(range(0, self.grid_size, 10))
        
        self.ax.grid(True)


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
                goal_handle, _  = self.ax.plot(*goal, 'rs', markersize=ms, label="Goal")
                agent_handle, _ = self.ax.plot(*agent.position, 'g^', markersize=ms, label="Agent")
            elif self.dim == 3:
                goal_handle  = self.ax.scatter(*goal, 'rs', c='red', marker='o', s=100, label="Goal")
                agent_handle = self.ax.scatter(*agent.position, c='green', marker='o', s=50, label="Agent")
        
        for obs in self.obstacles:
            if self.dim == 2:
                obstacle_handle, _ = self.ax.plot(*obs, 'k8', markersize=ms, label="Obstacle")
            elif self.dim == 3:
                obstacle_handle = self.ax.scatter(*obs, 'k8', c='black', marker='o', s=100, label="Obstacle")
        
        self.ax.legend([agent_handle, obstacle_handle, goal_handle], ['Agent', 'Obstacle', 'Goal'])
        self.fig.suptitle(title, fontsize=20)
        
        if interactive:
            plt.pause(waiting_time)  # Pause a bit so that updates are visible
            plt.draw()
        else:
            self.fig.savefig(osp.join(plotdir, f"{self.curr_step:03d}.png"))
            self.curr_step += 1
            
        
