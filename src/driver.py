import shutil
from torch.distributions import Categorical
import torch
import torch.nn.functional as F
import yaml

from .net import Actor, Critic
from .gnn import UniMPGNN, construct_graph
from .env import Environment

from torch_geometric.loader import DataLoader

import os.path as osp
import os
import cv2

class Driver():
    def __init__(self, config, lr=1e-3):
        self.env = Environment(config=config)
        
        env = self.env
        self.actor = Actor(
            env.agg_out_channels + 3 * env.dim, # agg dimension + (pos, vel, rel_goal) * self.dim
            len(env.action_to_move)
        )
        self.actor_gnn = UniMPGNN(
            3 * env.dim + 3,  # (rel_pos, vel, rel_goal) * self.dim + entity embedding
            env.agg_out_channels,
            aggregate=False
        )
        
        self.critic = Critic(
            env.agg_out_channels
        )
        self.critic_gnn = UniMPGNN(
            3 * env.dim + 3,  # (rel_pos, vel, rel_goal) * self.dim + entity embedding
            env.agg_out_channels,
            aggregate=True
        )
        
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()) + 
            list(self.actor_gnn.parameters()) + list(self.critic_gnn.parameters()),
            lr=lr
        )
        
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
        env = self.env
        loader = DataLoader([
            construct_graph(
                i, env.agents, env.goals, env.obstacles, env.sensing_radius, dim=env.dim
            )
            for i in range(env.num_agents)
        ], batch_size=env.num_agents)
        batch = next(iter(loader))
        
        actor_x_aggs = self.actor_gnn(batch)
        critic_X_aggs = self.critic_gnn(batch)
        
        return actor_x_aggs, critic_X_aggs
    
    
    def train(self, num_episodes, num_steps, run_name, gamma=0.8, 
                randomize_every=100, eval_every=1000, 
                save_models=True, save_every=1000, randomize=False,
                checkpoint=None):
        
        if checkpoint is not None:
            self.load_models(checkpoint)
        
        env = self.env
        for episode in range(num_episodes):
            # reset environment some times
            if randomize and (episode+1) % randomize_every == 0:
                env.reset(randomize=True)  
            else:
                env.reset()  
            
            # store episode data
            log_probs = []
            values = []
            rewards = []
            masks = []

            for step in range(num_steps):
                observations = env.get_observations()
                actor_x_aggs, critic_X_aggs = self.get_aggregated_information()
                
                # N * [obs(i), x_agg(i)]
                actor_inputs = torch.cat((observations, actor_x_aggs), dim=1)
                critic_inputs = critic_X_aggs
                
                # get action probabilities and take one action per agent (argmax)
                action_logits = self.actor(actor_inputs)
                dist = Categorical(logits=action_logits)
                actions = dist.mode
                log_prob = dist.log_prob(actions)
            
                # get state values 
                state_values = self.critic(critic_inputs)
        
                reward, dones, _ = env.step(actions) 
                
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
                R = rewards[step] + gamma * R * masks[step]
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
                print(f"Episode: {episode+1}, Loss: {loss.item():.3f}")
                self.eval(
                    num_steps=num_steps,
                    run_name=run_name,
                    randomize=randomize, 
                    render=False, 
                    load=False
                )
            
            if save_models and (episode+1) % save_every == 0:
                self.save_models(run_name)

    
    def eval(self, num_steps, run_name, randomize=False, render=False, load=False):
        env = self.env
        
        if load:
            self.load_models(run_name)

        if render:
            # save plots images in a tmp dir
            plotdir = "tmp"
            if not os.path.exists(plotdir):
                os.mkdir(plotdir)

        # reset the environment with randomized settings (new agents positions)
        env.reset(randomize=randomize)
        
        episode_collisions = 0
        total_rewards = 0
        
        for s in range(num_steps):
            observations = env.get_observations()
            actor_x_aggs, _ = self.get_aggregated_information()
            
            actor_inputs = torch.cat((observations, actor_x_aggs), dim=1)
            action_logits = self.actor(actor_inputs)
            dist = Categorical(logits=action_logits)
            actions = dist.mode
            
            rewards, dones, collisions = env.step(actions)
            
            total_rewards += rewards.sum().item()
            episode_collisions += collisions
            
            if render:
                env.render(plotdir=plotdir, title=f"Step: {s+1}/{num_steps}\n" + 
                                f"Total collisions: {episode_collisions} -- Done: {int(dones.sum())}/{env.num_agents} agents")
        
        print(f"\tEpisode reward: {total_rewards:.3f} -- Collisions: {episode_collisions} -- Done: {int(dones.sum())}/{env.num_agents} agents")
        
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
        with open(osp.join(model_dir, 'config.yaml'), 'w') as file:
            yaml.dump(self.env.config, file)
        
        
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

