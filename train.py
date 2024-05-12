from src.env import Environment
from src.driver import Driver
import glob


def setup_rundir():
    runs = sorted(glob.glob("runs/*"))
    if len(runs) > 0:
        run_name = f"run_{int(runs[-1].split('_')[1])+1:03d}"
    else:
        run_name = "run_001"
    
    return run_name


if __name__ == "__main__":
    
    # env params
    num_agents       = 7
    grid_size        = 30
    sensing_radius   = 3
    max_obstacles    = 20
    dim              = 2
    goal_reward      = 5
    collision_reward = -5
    agg_out_channels = 32
    
    config = {
        'num_agents': num_agents,
        'grid_size': grid_size,
        'sensing_radius': sensing_radius,
        'max_obstacles': max_obstacles,
        'goal_reward': goal_reward,
        'collision_reward': collision_reward,
        'dim': dim,
        'agg_out_channels': agg_out_channels,
    }
    
    lr = 1e-4
    
    driver = Driver(
        config=config,
        lr=lr
    )
    
    # training params
    episodes        = 10000
    steps           = int(grid_size * 1.5) # set a number of step that allows agents to reach goal 
    randomize_every = episodes // 20
    eval_every      = episodes // 20
    save_models     = True
    save_every      = episodes // 5
    gamma           = 0.8
    
    config["steps"] = steps
        
    run_name = setup_rundir()
    
    driver.train(
        num_episodes=episodes, 
        num_steps=steps, 
        run_name=run_name, 
        randomize_every=randomize_every, 
        eval_every=eval_every,
        save_models=save_models,
        save_every=save_every,
        randomize=True,
        gamma=gamma
    )
    
    driver.eval(
        num_steps=steps,
        run_name=run_name,
        render=True,
        load=False,
        randomize=True
    )
    
    # eval(env)
    