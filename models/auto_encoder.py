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
        
class encoder(nn.Module):
    """ Model for Encoder """

    def __init__(self, input_len = 1, z_len = 2048):
        super(encoder, self).__init__()
        
        self.cnn_l1  =  nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=(4, 4), stride = 2, padding = 1),
            nn.ReLU()
        )
        
        self.cnn_l2  =  nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(5, 5), stride = 2, padding = 1),
            nn.ReLU()
        )

        self.cnn_l3  =  nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=(5, 5), stride = 2, padding = 1),
            nn.ReLU()
        )
        self.cnn_l4  =  nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(4, 4), stride = 2),
            nn.ReLU()
        )
        self.fc1 = nn.Sequential(
            torch.nn.Linear(32 * 14 * 14, 4096),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            torch.nn.Linear(4096, z_len)
        )
        
    def forward(self, input):
#         print(input.size())
        x = self.cnn_l1(input)
#         print(x.size())
        x = self.cnn_l2(x)
#         print(x.size())
        x = self.cnn_l3(x)
#         print(x.size())
        x = self.cnn_l4(x)
#         print(x.size())
        x = self.fc1(x.view(x.size()[0], -1))
#         print(x.size())
        return self.fc2(x)
    
class decoder(nn.Module):
    """ Model for Decoder """

    def __init__(self, z_len = 2048 + 4, output_len = 1):
        super(decoder, self).__init__()
        
        self.fc1 = nn.Sequential(
            torch.nn.Linear(z_len, 4096),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            torch.nn.Linear(4096, 32 * 14 * 14),
            nn.ReLU()
        )
        self.cnntp_l1  =  nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=(4, 4), stride = 2, output_padding = 1),
            nn.ReLU()
        )
        self.cnntp_l2  =  nn.Sequential(
            nn.ConvTranspose2d(32, 64, kernel_size=(5, 5), stride = 2, padding = 1),
            nn.ReLU()
        )
        self.cnntp_l3  =  nn.Sequential(
            nn.ConvTranspose2d(64, 128, kernel_size=(5, 5), stride = 2, padding = 1),
#             nn.ReLU()
        )
        self.cnntp_l4  =  nn.Sequential(
            nn.ConvTranspose2d(128, output_len, kernel_size=(4, 4), stride = 2),
#             nn.ReLU()
#             nn.Sigmoid()
        )
        
#         self.cnntp_l5  =  nn.Sequential(
#             nn.ConvTranspose2d(128, 128, kernel_size=(3, 3), padding = 1),
# #             nn.Sigmoid()
#         )
    def forward(self, input):
        x = self.fc1(input)
        x = self.fc2(x)
#         print(x.size())
        x = self.cnntp_l1(x.view(x.size()[0], 32, 14, 14))
#         print(x.size())
        x = self.cnntp_l2(x)
#         print(x.size())
        x = self.cnntp_l3(x)
#         print(x.size())
#         x = self.cnntp_l4(x)
#         return self.cnntp_l5(x)
        return self.cnntp_l4(x)
    
class _AEModel(nn.Module):
    def __init__(self, input_len, z_len, action_len = 4):
        super(_AEModel, self).__init__()
        
        self.emodel = encoder(input_len, z_len)
        z_len += action_len
        self.dmodel = decoder(z_len, input_len)

        if use_cuda:
            print("Using GPU")
            self.emodel = self.emodel.cuda()
            self.dmodel = self.dmodel.cuda()
        else:
            print("Using CPU")
            
    def forward(self, input, actions):
        z = self.emodel(input)
#         print(z)
        z = torch.cat((z, actions), 1)
#         print(z)
        return self.dmodel(z)
    
class AEModel():
    def __init__(self, input_len, z_len, action_len = 4, learning_rate = 0.0001):

        self.aemodel = _AEModel(input_len, z_len, action_len = action_len)
        if use_cuda:
            print("Using GPU")
            self.aemodel = self.aemodel.cuda()
        else:
            print("Using CPU")
#         self.model = nn.DataParallel(self.model)
        self.optimizer = Adam(self.aemodel.parameters(), lr = learning_rate)
#         self.loss_fn = nn.SmoothL1Loss()
        self.loss_fn = nn.MSELoss()
        
    def predict(self, input, action):
        input = FloatTensor(input).unsqueeze(0)
        action = FloatTensor(action).unsqueeze(0)
        image = self.aemodel(input, action)

        return image

    def predict_batch(self, input, actions):
        input = FloatTensor(input)
        actions = FloatTensor(actions)
        images = self.aemodel(input, actions)
        
        return images

    def fit(self, predict, ground_truth):
#         print(predict.size())
#         print(ground_truth.size())
        loss = self.loss_fn(predict, ground_truth)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def loss_only(self, predict, ground_truth):
#         print(predict.size())
#         print(ground_truth.size())
        loss = self.loss_fn(predict, ground_truth)
        return loss
    
    def save(self, path):
        torch.save(self.aemodel.state_dict(), path)
        
    def load(self, path):
        self.aemodel.load_state_dict(torch.load(path))