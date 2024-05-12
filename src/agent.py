import torch

class Agent:
    def __init__(self, start_pos, grid_size, dim=2):
        self.position = start_pos
        self.velocity = torch.zeros(dim)
        self.grid_size = grid_size
        
    def move(self, move):
        move = torch.tensor(move, dtype=torch.float)
        self.position += move
        self.position = torch.clamp(self.position, 0, self.grid_size - 1)
        self.velocity = move
