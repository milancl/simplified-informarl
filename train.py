from src.env import Environment
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
    num_agents     = 7
    grid_size      = 25
    sensing_radius = 3
    max_obstacles  = 20
    dim = 3
    
    # training params
    episodes        = 100000
    steps           = int(grid_size * 1.5) # set a number of step that allows agents to reach goal 
    randomize_every = episodes // 20
    eval_every      = episodes // 20
    save_models     = True
    save_every      = episodes // 5
    
    env = Environment(
        num_agents=num_agents, 
        grid_size=grid_size,
        sensing_radius=sensing_radius,
        dim=dim
    )
    
    run_name = setup_rundir()
    
    env.train(
        num_episodes=episodes, 
        num_steps=steps, 
        run_name=run_name, 
        randomize_every=randomize_every, 
        eval_every=eval_every,
        save_models=save_models,
        save_every=save_every,
        randomize=True
    )
    
    env.eval(
        num_steps=steps,
        run_name=run_name,
        render=True,
        load=False,
        randomize=True
    )
    
    # eval(env)
    