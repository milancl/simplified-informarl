from src.env import Environment    
    
if __name__ == "__main__":
    
    # env params
    num_agents     = 7
    grid_size      = 25
    sensing_radius = 3
    max_obstacles  = 20
    dim = 3
    
    steps           = grid_size * 2
    
    env = Environment(
        num_agents=num_agents, 
        grid_size=grid_size,
        sensing_radius=sensing_radius,
        max_obstacles=max_obstacles,
        dim=dim
    )
    
    run_name = "run_005"
    
    env.eval(
        num_steps=steps, 
        run_name=run_name, 
        render=True, 
        load=True
    )
    