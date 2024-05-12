import yaml
from src.runner import GraphRunner, MLPRunner
import glob

import os.path as osp
from sys import argv


def setup_rundir():
    runs = sorted(glob.glob("runs/*"))
    if len(runs) > 0:
        run_name = f"run_{int(runs[-1].split('_')[1])+1:03d}"
    else:
        run_name = "run_001"
    
    return run_name


if __name__ == "__main__":
    
    checkpoint = None
    if len(argv) > 1:
        checkpoint = argv[1]    
    
    # env params
    if checkpoint is None:
    
        num_agents       = 7
        grid_size        = 25
        sensing_radius   = 3
        max_obstacles    = 20
        dim              = 2
        goal_reward      = 5
        collision_reward = -5
        agg_out_channels = 32
        n_random_initializations = 30
        
        config = {
            'num_agents': num_agents,
            'grid_size': grid_size,
            'sensing_radius': sensing_radius,
            'max_obstacles': max_obstacles,
            'goal_reward': goal_reward,
            'collision_reward': collision_reward,
            'dim': dim,
            'agg_out_channels': agg_out_channels,
            'n_random_initializations': n_random_initializations
        }
    else:
        with open(osp.join("runs", checkpoint, "config.yaml"), 'r') as file:
            config = yaml.safe_load(file)
            
    grid_size = config["grid_size"]
    
    actor_lr = 1e-5
    critic_lr = 1e-5
    
    runner = GraphRunner(
        config=config,
        actor_lr=actor_lr, 
        critic_lr=critic_lr,
    )
    
    # training params
    episodes        = 100
    steps           = int(grid_size * 1.5) # set a number of step that allows agents to reach goal 
    randomize_every = episodes // 50
    eval_every      = episodes // 30
    train_log_every = episodes // 30
    save_models     = True
    save_every      = episodes // 5
    gamma           = 0.8
    
    config["steps"] = steps
        
    run_name = setup_rundir()
    
    runner.train(
        num_episodes=episodes, 
        num_steps=steps, 
        run_name=run_name, 
        randomize_every=randomize_every, 
        eval_every=eval_every,
        save_models=save_models,
        save_every=save_every,
        randomize=True,
        gamma=gamma,
        checkpoint=checkpoint,
        train_log_every=train_log_every
    )
    
    runner.eval(
        num_steps=steps,
        run_name=run_name,
        render=True,
        load=False,
    )
    
    # eval(env)
    