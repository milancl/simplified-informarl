import torch
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, global_mean_pool
import torch.nn.functional as F
from torch_geometric.utils import softmax
from torch import nn

def entity_one_hot(type):
    if type == 'agent':
        return torch.tensor([1, 0, 0], dtype=torch.float)
    elif type == 'goal':
        return torch.tensor([0, 1, 0], dtype=torch.float)
    elif type == 'obstacle':
        return torch.tensor([0, 0, 1], dtype=torch.float)
    else:
        raise ValueError("Entity type does not exist")

def grid_distance(rel_pos):
    return torch.max(torch.abs(rel_pos))

def construct_graph(i, agents, goals, obstacles, sensing_radius, dim=2):    
    agent_i = agents[i]
    
    agents_features = torch.stack([
        torch.cat((
            agent_j.position - agent_i.position, agent_j.position, goal_j - agent_i.position, entity_one_hot('agent')
        ))
        for agent_j, goal_j in zip(agents, goals)
    ])

    # for agent_j, goal_j in zip(agents, goals):
    #     rel_pos = agent_j.position - agent_i.position
    #     vel_j = agent_j.position
    #     rel_goal = goal_j - agent_i.position
    #     x = np.concatenate((rel_pos, vel_j, rel_goal, entity_one_hot('agent')))
    #     X.append(x)
    
    goal_features = torch.stack([
        torch.cat((
            goal - agent_i.position, torch.zeros(dim), goal - agent_i.position, entity_one_hot('goal')
        ))
        for goal in goals
    ])
    
    # for goal in goals:
    #     rel_pos = goal - agent_i.position
    #     vel_j = np.zeros(2)
    #     rel_goal = rel_pos
    #     x = np.concatenate((rel_pos, vel_j, rel_goal, entity_one_hot('goal')))
    #     X.append(x)
    
    if len(obstacles) > 0:
        obs_features = torch.stack([
            torch.cat((
                obs - agent_i.position, torch.zeros(dim), obs - agent_i.position, entity_one_hot('obstacle')
            ))
            for obs in obstacles
        ])
        
    # for obs in obstacles:
    #     rel_pos = obs - agent_i.position
    #     vel_j = np.zeros(2)
    #     rel_goal = rel_pos
    #     x = np.concatenate((rel_pos, vel_j, rel_goal, entity_one_hot('obstacle')))
    #     X.append(x)
    
    if len(obstacles) > 0:
        X = torch.cat((
            agents_features, goal_features, obs_features
        ), dim=0)
    else:
        X = torch.cat((
            agents_features, goal_features
        ), dim=0)
        
    edge_index = []
    edge_attr = []    
    
    for i in range(len(agents)):
        for j in range(i+1, X.size(0)):
            agent = X[i]
            entity = X[j]
            
            vec = agent[:2] - entity[:2] # distance between agent i and entity j
            dist = grid_distance(vec)
            if dist <= sensing_radius:
                if j >= len(agents): # j is not an agent (edge between non-agent to agent)
                    edge_index.append([j, i])
                    edge_attr.append([dist])
                else: # agant-agent bidirectional edge
                    edge_index.append([i, j])
                    edge_index.append([j, i])
                    edge_attr.append([dist])
                    edge_attr.append([dist])
        
    if len(edge_index) == 0:
        # ensure edge_index is 2D with zero columns if there are no edges
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    return Data(x=X, edge_index=edge_index, edge_attr=edge_attr)


class UniMPLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1):
        super(UniMPLayer, self).__init__(aggr='add')  # Use 'add' for aggregating messages.
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        
        # Define the weights for the linear transformations
        self.w1 = nn.Linear(in_channels, out_channels)
        self.w2 = nn.Linear(in_channels, out_channels)
        self.w3 = nn.Linear(in_channels, out_channels)
        self.w4 = nn.Linear(in_channels, out_channels)
        self.w5 = nn.Linear(1, out_channels)  # Assuming edge features are 1-dimensional

        self.sqrt_d = torch.sqrt(torch.tensor(out_channels, dtype=torch.float))

    def forward(self, x, edge_index, edge_attr):
        # Start propagating messages
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_index_i, edge_attr, size_i):
        # Transform node features and edge features
        x_i_transformed = self.w3(x_i)
        x_j_transformed = self.w4(x_j)
        edge_attr_transformed = self.w5(edge_attr)

        # Compute attention scores
        scores = (x_i_transformed * (x_j_transformed + edge_attr_transformed)).sum(dim=-1)
        scores = scores / self.sqrt_d
        alpha = softmax(scores, edge_index_i, num_nodes=size_i)
        
        # Transform node features for message passing
        return alpha.view(-1, 1) * self.w2(x_j)

    def update(self, aggr_out, x):
        # Node updates based on its features and aggregated messages
        return self.w1(x) + aggr_out


class UniMPGNN(nn.Module):
    def __init__(self, in_channels, out_channels, aggregate, heads=1):
        super(UniMPGNN, self).__init__()
        self.layer = UniMPLayer(in_channels, out_channels, heads)
        self.aggregate = aggregate

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        node_out = self.layer(x, edge_index, edge_attr)
        if not self.aggregate:
            # take the ith vector x_agg from the ith graph
            num_agents = data.num_graphs
            num_entities = data.num_nodes // num_agents
            node_out = node_out.view(num_agents, num_entities, -1)

            idx = torch.arange(num_agents)
            
            return node_out[idx, idx]
        else:
            return global_mean_pool(node_out, data.batch)

