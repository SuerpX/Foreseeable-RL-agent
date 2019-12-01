import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

def weights_initialize(module):
    if type(module) == nn.Linear:
        nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('tanh'))
        module.bias.data.fill_(0.01)

class dqn(nn.Module):
    """ Model for Decoder """

    def __init__(self, z_len = 2048 + 4, output_len = 1):
        super(dqn, self).__init__()
        
        self.fc1 = nn.Sequential(
            torch.nn.Linear(z_len, 512),
            nn.ReLU()
        )
        weights_initialize(self.fc1)
        self.fc2 = nn.Sequential(
            torch.nn.Linear(512, output_len),
            nn.ReLU()
        )
        weights_initialize(self.fc2)
        
    def forward(self, input):
        x = self.fc1(input)
        x = self.fc2(x)
        return x
    
class _EDQNModel(nn.Module):
    def __init__(self, encoder, z_len, output_len = 1, action_len = 4):
        super(_EDQNModel, self).__init__()
        
        self.emodel = encoder
        z_len += action_len
        self.dqnmodel = dqn(z_len, output_len)
        self.action_len = action_len
        if use_cuda:
            print("Using GPU")
            self.emodel = self.emodel.cuda()
            self.dqnmodel = self.dqnmodel.cuda()
        else:
            print("Using CPU")
            
    def forward(self, input, actions):
        z = self.emodel(input)
#         print(z)
        if actions.size()[0] / input.size()[0] == self.action_len:
            z = torch.cat(self.action_len * [z])
        z = torch.cat((z, actions), 1)
#         print(z)
        return self.dqnmodel(z)
    
class EDQNModel():
    def __init__(self, z_len, encoder, output_len, action_len = 4, learning_rate = 0.0001):

        self.edqnmodel = _EDQNModel(encoder, z_len, output_len = output_len, action_len = action_len)
        if use_cuda:
            print("Using GPU")
            self.edqnmodel = self.edqnmodel.cuda()
        else:
            print("Using CPU")
#         self.model = nn.DataParallel(self.model)
        self.optimizer = Adam(self.edqnmodel.parameters(), lr = learning_rate)
#         self.loss_fn = nn.SmoothL1Loss()
        self.loss_fn = nn.MSELoss()
        
    def predict(self, input):
        input = FloatTensor(input).unsqueeze(0)
        actions = FloatTensor(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))
        q_values = self.edqnmodel(input, actions)
        
        a = torch.argmax(q_values.view(-1))
#         print(a,q_values.view(-1) )

        return a.item(), q_values

    def predict_batch(self, input, actions):
        input = FloatTensor(input)
        actions = FloatTensor(actions)
        q_values = self.edqnmodel(input, actions)
        
        return q_values
    
    def fit(self, predict_q_value, target_q_value):
#         print(predict.size())
#         print(ground_truth.size())
        loss = self.loss_fn(predict_q_value, target_q_value)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def loss_only(self, predict_q_value, target_q_value):
#         print(predict.size())
#         print(ground_truth.size())
        loss = self.loss_fn(predict_q_value, target_q_value)
        return loss
    
    def replace(self, dest):
        self.edqnmodel.load_state_dict(dest.edqnmodel.state_dict())
    
    def save(self, path):
        torch.save(self.edqnmodel.state_dict(), path)
        
    def load(self, path):
        self.edqnmodel.load_state_dict(torch.load(path))
    
    def replace_soft(self, dest, tau = 1e-3):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, eval_param in zip(self.edqnmodel.parameters(), dest.edqnmodel.parameters()):
            target_param.data.copy_(tau*eval_param.data + (1.0-tau)*target_param.data)
