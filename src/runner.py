import shutil
from torch.distributions import Categorical
import torch
import torch.nn.functional as F
import yaml

from torch_geometric.nn import global_mean_pool

from .net import Actor, Critic, SimpleMLP
from .gnn import UniMPGNN, construct_graph
from .env import Environment

from torch_geometric.loader import DataLoader

import os.path as osp
import os
import cv2

import pickle

class BaseRunner():
    def __init__(self, config):
        self.env = Environment(config=config)
        
        self.history = {
            "train": [],
            "eval": []
        }
        
        env = self.env
        self.actor = Actor(
            env.agg_out_channels + 3 * env.dim, # agg dimension + (pos, vel, rel_goal) * self.dim
            len(env.action_to_move)
        )
        
        self.critic = Critic(
            env.agg_out_channels
        )
        
    def get_graph_batch(self):
        # Returns two matrices N * c, where N is the number of agents and 
        # c is the aggregated vector dimension. The first corresponds to 
        # the ith vectors in the iths graph. The second correcponds to each
        # aggregated vector for each graph.
        
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
        
        return batch
    
                
    def save_history(self, run_name):
        run_dir = osp.join("runs", run_name)
        with open(osp.join(run_dir, "history.pickle"), "wb") as f:
            pickle.dump(self.history, f)
        
        
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




class GraphRunner(BaseRunner):
    def __init__(self, config, actor_lr=1e-3, critic_lr=1e-3):
        super().__init__(config)
        
        self.actor_gnn = UniMPGNN(
            3 * self.env.dim + 3,  # (rel_pos, vel, rel_goal) * self.dim + entity embedding
            self.env.agg_out_channels,
            aggregate=False
        )
 
        self.critic_gnn = UniMPGNN(
            3 * self.env.dim + 3,  # (rel_pos, vel, rel_goal) * self.dim + entity embedding
            self.env.agg_out_channels,
            aggregate=True
        )
        
        self.optimizer_actor = torch.optim.Adam(
            list(self.actor.parameters()) +
            list(self.actor_gnn.parameters()),
            lr=actor_lr
        )
        
        self.optimizer_critic = torch.optim.Adam(
            list(self.critic.parameters()) + 
            list(self.critic_gnn.parameters()),
            lr=critic_lr
        )
    
    def train(self, num_episodes, num_steps, run_name, gamma=0.8, 
                randomize_every=100, eval_every=1000, 
                save_models=True, save_every=1000, randomize=False,
                checkpoint=None, train_log_every=100):
        
        self.actor.train()
        self.critic.train()
        self.actor_gnn.train()
        self.critic_gnn.train()
        
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
            # masks = []
            
            for step in range(num_steps):
                observations = env.get_observations()
                
                batch = self.get_graph_batch()
                actor_x_aggs = self.actor_gnn(batch)
                critic_X_aggs = self.critic_gnn(batch)
                
                # N * [obs(i), x_agg(i)]
                actor_inputs = torch.cat((observations, actor_x_aggs), dim=1)
                critic_inputs = critic_X_aggs
                
                # get action probabilities and take one action per agent (argmax)
                action_logits = self.actor(actor_inputs)
                dist = Categorical(logits=action_logits)
                actions = dist.sample()
                log_prob = dist.log_prob(actions)
            
                # get state values 
                state_values = self.critic(critic_inputs)
        
                reward, _, _ = env.step(actions) 
                
                log_probs.append(log_prob)
                values.append(state_values)
                rewards.append(reward.unsqueeze(1))
                # masks.append(1 - dones.unsqueeze(1))
            
            # convert to tensors
            log_probs = torch.cat(log_probs)
            values = torch.cat(values) 
            rewards = torch.cat(rewards)
            # masks = torch.cat(masks)

            # compute returns (start from the last to compute the correct gamma weighting)
            returns = []
            R = 0
            for step in reversed(range(len(rewards))):
                # R = rewards[step] + gamma * R * masks[step]
                R = rewards[step] + gamma * R
                returns.insert(0, R)

            # Advantages are computed as the expected critic reward, and the actual 
            # return computed in the environment. If an advantage is positive, it increases 
            # the loss, meaning that a good action was taken. If negative, they decrease the 
            # loss, meaning a bad action was chosen. Here the loss is steepest ascent.
            returns = torch.cat(returns)
            advantages = returns - values

            # losses
            actor_loss = -(log_probs * advantages.detach()).mean()
            # values and returns should have the same shape
            critic_loss = F.mse_loss(values.squeeze(), returns.detach())
            
            # backprop
            self.optimizer_actor.zero_grad()
            self.optimizer_critic.zero_grad()
            
            actor_loss.backward()
            self.optimizer_actor.step()
            
            critic_loss.backward()
            self.optimizer_critic.step()
            
            # log and save models
            if (episode+1) % train_log_every == 0:
                self.history["train"].append({
                    "steps": (episode + 1) * num_steps,
                    "actor_loss": actor_loss.item(),
                    "critic_loss": critic_loss.item(),
                    "total_episode_reward": rewards.sum().item(),
                    "average_episode_reward": rewards.sum(0).mean().item()
                })
            
            if (episode+1) % eval_every == 0:
                print(f"Episode: {episode+1}, Value loss: {critic_loss.item():.3f}, Policy loss: {actor_loss.item():.3f}")
                self.eval(
                    num_steps=num_steps,
                    run_name=run_name,
                    render=False, 
                    load=False,
                    episode=episode
                )
                self.actor.train()
                self.critic.train()
                self.actor_gnn.train()
                self.critic_gnn.train()
                
            
            if save_models and (episode+1) % save_every == 0:
                self.save_models(run_name)
                self.save_history(run_name)

    
    def eval(self, num_steps, run_name, render=False, load=False, episode=0):
        env = self.env
        
        self.actor.eval()
        self.critic.eval()
        self.actor_gnn.eval()
        self.critic_gnn.eval()
        
        if load:
            self.load_models(run_name)

        if render:
            # save plots images in a tmp dir
            plotdir = "tmp"
            if not os.path.exists(plotdir):
                os.mkdir(plotdir)

        # reset the environment with eval settings (new agents positions)
        env.reset(eval=True)
        
        episode_collisions = 0
        total_rewards = 0
        
        for s in range(num_steps):
            observations = env.get_observations()
            
            batch = self.get_graph_batch()
            actor_x_aggs = self.actor_gnn(batch)
            
            actor_inputs = torch.cat((observations, actor_x_aggs), dim=1)
            action_logits = self.actor(actor_inputs)
            dist = Categorical(logits=action_logits)
            actions = dist.sample()
            
            rewards, dones, collisions = env.step(actions)
            
            total_rewards += rewards.sum().item()
            episode_collisions += collisions
            
            if render:
                env.render(plotdir=plotdir, title=f"Step: {s+1}/{num_steps}\n" + 
                                f"Total collisions: {episode_collisions} -- Done: {int(dones.sum())}/{env.num_agents} agents")
        
        print(f"\tEpisode reward: {total_rewards:.3f} -- Collisions: {episode_collisions} -- Done: {int(dones.sum())}/{env.num_agents} agents")
        
        self.history["eval"].append({
            "steps": (episode+1) * num_steps,
            "total_episode_reward": total_rewards,
            "average_episode_reward": rewards.sum(0).mean().item()
        })
        
        if render:
            # If rendering, perhaps convert saved images to a video or similar
            self.save_video(run_name, plotdir, fps=6)  # Make sure you have a function `save_video`
            # Clean up after rendering
            shutil.rmtree(plotdir)
            
            
    def load_models(self, run_name):
        model_dir = osp.join("runs", run_name)
        assert osp.exists(model_dir)
        
        self.actor.load_state_dict(torch.load(osp.join(model_dir, "models", "actor.pth")))
        self.critic.load_state_dict(torch.load(osp.join(model_dir, "models", "critic.pth")))
        self.actor_gnn.load_state_dict(torch.load(osp.join(model_dir, "models", "actor_gnn.pth")))
        self.critic_gnn.load_state_dict(torch.load(osp.join(model_dir, "models", "critic_gnn.pth")))
        
        self.actor.eval()
        self.critic.eval()
        self.actor_gnn.eval()
        self.critic_gnn.eval()


    def save_models(self, run_name):
        model_dir = osp.join("runs", run_name)
        if not osp.isdir(model_dir):
            os.mkdir(model_dir)
            os.mkdir(osp.join(model_dir, "models"))
        torch.save(self.actor.state_dict(), osp.join(model_dir, "models", "actor.pth"))
        torch.save(self.critic.state_dict(), osp.join(model_dir, "models", "critic.pth"))
        torch.save(self.actor_gnn.state_dict(), osp.join(model_dir, "models", "actor_gnn.pth"))
        torch.save(self.actor_gnn.state_dict(), osp.join(model_dir, "models", "critic_gnn.pth"))
        with open(osp.join(model_dir, 'config.yaml'), 'w') as file:
            yaml.dump(self.env.config, file)


class MLPRunner(BaseRunner):
    def __init__(self, config, actor_lr=1e-3, critic_lr=1e-3):
        super().__init__(config)
        
        env = self.env
        self.actor_mlp = SimpleMLP(
            3 * self.env.dim,
            env.agg_out_channels + 3 * env.dim
        )
        
        self.critic_mlp = SimpleMLP(
            3 * self.env.dim + 3,
            env.agg_out_channels
        )
        
        self.optimizer_actor = torch.optim.Adam(
            list(self.actor.parameters()) + 
            list(self.actor_mlp.parameters()),
            lr=actor_lr
        )
        
        self.optimizer_critic = torch.optim.Adam(
            list(self.critic.parameters()) +
            list(self.critic_mlp.parameters()),
            lr=critic_lr
        )
    
    def train(self, num_episodes, num_steps, run_name, gamma=0.8, 
                randomize_every=100, eval_every=1000, 
                save_models=True, save_every=1000, randomize=False,
                checkpoint=None, train_log_every=100):
        
        self.actor.train()
        self.critic.train()
        self.actor_mlp.train()
        self.critic_mlp.train()
        
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
            # masks = []
            
            tot_rewards = 0
            
            for step in range(num_steps):
                observations = env.get_observations()
                actor_inputs = self.actor_mlp(observations)
                
                batch = self.get_graph_batch()
                critic_inputs = self.critic_mlp(global_mean_pool(batch.x, batch.batch))
                
                # get action probabilities and take one action per agent 
                action_logits = self.actor(actor_inputs)
                dist = Categorical(logits=action_logits)
                actions = dist.sample()
                log_prob = dist.log_prob(actions)
            
                # get state values 
                state_values = self.critic(critic_inputs)
        
                reward, _, _ = env.step(actions) 
                
                tot_rewards += reward.sum().item()
                
                log_probs.append(log_prob)
                values.append(state_values)
                rewards.append(reward.unsqueeze(1))
                # masks.append(1 - dones.unsqueeze(1))
            
            # convert to tensors
            log_probs = torch.cat(log_probs)
            values = torch.cat(values) 
            rewards = torch.cat(rewards)
            # masks = torch.cat(masks)

            # compute returns (start from the last to compute the correct gamma weighting)
            returns = []
            R = 0
            for step in reversed(range(len(rewards))):
                # R = rewards[step] + gamma * R * masks[step]
                R = rewards[step] + gamma * R
                returns.insert(0, R)

            # Advantages are computed as the expected critic reward, and the actual 
            # return computed in the environment. If an advantage is positive, it increases 
            # the loss, meaning that a good action was taken. If negative, they decrease the 
            # loss, meaning a bad action was chosen. Here the loss is steepest ascent.
            returns = torch.cat(returns)
            advantages = returns - values

            # losses
            actor_loss = -(log_probs * advantages.detach()).mean()
            # values and returns should have the same shape
            critic_loss = F.mse_loss(values.squeeze(), returns.detach())
            
            # backprop
            self.optimizer_actor.zero_grad()
            self.optimizer_critic.zero_grad()
            
            actor_loss.backward()
            self.optimizer_actor.step()
            
            critic_loss.backward()
            self.optimizer_critic.step()
            
            # log and save models
            if (episode+1) % train_log_every == 0:
                self.history["train"].append({
                    "steps": (episode + 1) * num_steps,
                    "actor_loss": actor_loss.item(),
                    "critic_loss": critic_loss.item(),
                    "total_episode_reward": tot_rewards,
                    "average_episode_reward": rewards.sum(0).mean().item()
                })
            
            if (episode+1) % eval_every == 0:
                print(f"Episode: {episode+1}, Value loss: {critic_loss.item():.3f}, Policy loss: {actor_loss.item():.3f}")
                self.eval(
                    num_steps=num_steps,
                    run_name=run_name,
                    render=False, 
                    load=False,
                    episode=episode
                )
                self.actor.train()
                self.critic.train()
                self.actor_mlp.train()
                self.critic_mlp.train()
                
            
            if save_models and (episode+1) % save_every == 0:
                self.save_models(run_name)
                self.save_history(run_name)

    
    def eval(self, num_steps, run_name, render=False, load=False, episode=0):
        env = self.env
        
        self.actor.eval()
        self.critic.eval()
        self.actor_mlp.eval()
        self.critic_mlp.eval()
        
        if load:
            self.load_models(run_name)

        if render:
            # save plots images in a tmp dir
            plotdir = "tmp"
            if not os.path.exists(plotdir):
                os.mkdir(plotdir)

        # reset the environment with eval settings (new agents positions)
        env.reset(eval=True)
        
        episode_collisions = 0
        total_rewards = 0
        
        rewards = []
        
        for s in range(num_steps):
            observations = env.get_observations()
            actor_inputs = self.actor_mlp(observations)
            
            action_logits = self.actor(actor_inputs)
            dist = Categorical(logits=action_logits)
            actions = dist.sample()
            
            reward, dones, collisions = env.step(actions)
            
            rewards.append(reward)
            
            total_rewards += reward.sum().item()
            episode_collisions += collisions
            
            if render:
                env.render(plotdir=plotdir, title=f"Step: {s+1}/{num_steps}\n" + 
                                f"Total collisions: {episode_collisions} -- Done: {int(dones.sum())}/{env.num_agents} agents")
        
        rewards = torch.cat(rewards)
        
        print(f"\tEpisode reward: {total_rewards:.3f} -- Collisions: {episode_collisions} -- Done: {int(dones.sum())}/{env.num_agents} agents")
        
        self.history["eval"].append({
            "steps": (episode+1) * num_steps,
            "total_episode_reward": total_rewards,
            "average_episode_reward": rewards.sum(0).mean().item()
        })
        
        if render:
            # If rendering, perhaps convert saved images to a video or similar
            self.save_video(run_name, plotdir, fps=6)  # Make sure you have a function `save_video`
            # Clean up after rendering
            shutil.rmtree(plotdir)
            
            
    def load_models(self, run_name):
        model_dir = osp.join("runs", run_name)
        assert osp.exists(model_dir)
        
        self.actor.load_state_dict(torch.load(osp.join(model_dir, "models", "actor.pth")))
        self.critic.load_state_dict(torch.load(osp.join(model_dir, "models", "critic.pth")))
        self.actor_mlp.load_state_dict(torch.load(osp.join(model_dir, "models", "actor_mlp.pth")))
        self.critic_mlp.load_state_dict(torch.load(osp.join(model_dir, "models", "critic_mlp.pth")))


    def save_models(self, run_name):
        model_dir = osp.join("runs", run_name)
        if not osp.isdir(model_dir):
            os.mkdir(model_dir)
            os.mkdir(osp.join(model_dir, "models"))
        torch.save(self.actor.state_dict(), osp.join(model_dir, "models", "actor.pth"))
        torch.save(self.critic.state_dict(), osp.join(model_dir, "models", "critic.pth"))
        torch.save(self.actor_mlp.state_dict(), osp.join(model_dir, "models", "actor_mlp.pth"))
        torch.save(self.critic_mlp.state_dict(), osp.join(model_dir, "models", "critic_mlp.pth"))
        with open(osp.join(model_dir, 'config.yaml'), 'w') as file:
            yaml.dump(self.env.config, file)

