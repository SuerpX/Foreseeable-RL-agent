import time
from sys import maxsize
import torch
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import cv2
import os

from models.auto_encoder import AEModel, encoder
from models.encoder_dqn import EDQNModel 
from utils import Node

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ACTION_DICT = {
    "STOP": 0,
    "FORWARD": 1,
    "LEFT": 2,
    "RIGHT": 3
}

ACTION_DICT_REVERSE = {
    0 : "STOP",
    1 : "FORWARD",
    2 : "LEFT",
    3 : "RIGHT"
}
class MBTSAgent():
    def __init__(self, max_depth = 5, action_space = len(ACTION_DICT), finish_action = 0):
        self.action_space = action_space
        self.max_depth = max_depth
        self.finish_action = finish_action
        self.root = None
        self.last_action = None
        
        # read trainsition model and evaluation model
        with torch.no_grad():
            self.read_model()
            self.eval_mode()
    
    def read_model(self):
        self.ae_model = AEModel(1, 2048)
        self.dqn_model = EDQNModel(2048, self.ae_model.aemodel.emodel, 1)
        self.load_model_best_dqn()
        self.load_model_best_ae()
        
    def load_model_best_dqn(self):
        print("loaded best dqn model")
        self.dqn_model.load('save_models/best_model_dqn-Copy1.pt')
        
    def load_model_best_ae(self):
        print("loaded best ae model")
        self.ae_model.load('save_models/best_model_ae-Copy1.pt') 
        
    def eval_mode(self):
        self.ae_model.aemodel.eval()
        self.dqn_model.edqnmodel.eval()
    
    def reward_func(self):
        return -2.5
    def tree_search(self, current_state, last_action, parent = None, depth = 0):
        # (self, name, state, reward = 0, parent = None, parent_action = None, best_q_value = None):
        if parent is None:
            parent = Node("level_{}_action_{}".format(depth, None), current_state.detach().cpu().numpy())
        
        next_states, actions = self.get_next_states(current_state)
        actions = (actions == 1).nonzero()[:, 1]
#         print(actions)
#         print(next_states)

        # expand
        best_q = -maxsize
        best_a = None
        best_child = None
#         print(next_states.size(), actions.size())
        if depth < self.max_depth:
#             if depth != 0:
#                 tqdm.disable()
            for ns, a in tqdm(zip(next_states, actions), disable=os.environ.get("DISABLE_TQDM", depth != 0), total = len(actions)):
#                 if (last_action == 2 and a.item() == 3) or (last_action == 3 and a.item() == 2):
#                     continue
                action_name = ACTION_DICT_REVERSE[a.item()]
                child = Node("level_{}_action_{}".format(depth + 1, a), ns.detach().cpu().numpy(), parent = parent, action_name = action_name)
                best_q_value, _, _ = self.tree_search(ns, a, parent = child, depth = depth + 1)
                parent.add_child(child)
                best_q_value += self.reward_func()
#                 if depth == 0:
#                     print(best_q_value, best_action)
                if (last_action == 2 and a.item() == 3) or (last_action == 3 and a.item() == 2):
                    continue
                if best_q_value > best_q:
                    best_q = best_q_value
                    best_a = a.item()
                    best_child = child
#                     print(last_action, best_a)
  
            best_q_value, a = self.eval_finish_action(ns)
#             if depth == 0:
#                  print(best_q_value, best_action) 
            if best_q_value > best_q:
                best_q = best_q_value
                best_a = a
#             if depth == 1:
#                 print("finish", best_q_value)
#                 print("best", best_q)
        else:
            best_a, q_value = self.dqn_model.predict(current_state)
            best_q = q_value.max(0)[0]
#             print(best_a, best_q)
        parent.best_q_value = best_q
        parent.best_child = best_child
        parent.best_action = best_a
#         print(last_action, best_a)
#         print(best_a)

        return best_q, best_a, parent
    
    def eval_finish_action(self, ns):
        finish_action = np.zeros((1, self.action_space))
        finish_action[0, self.finish_action] = 1 
        input_state = ns.unsqueeze(0)
        q_value = self.dqn_model.predict_batch(input_state, finish_action)
        return q_value[0][0].item(), self.finish_action
    
    def get_next_states(self, current_state):
#         print(self.action_space, self.action_space)
        actions = FloatTensor(np.zeros((self.action_space - 1, self.action_space)))
        for i in range(self.action_space):
            if i != self.finish_action:
                actions[i - 1, i] = 1
        
#         print(actions)
        input_states = current_state.repeat(actions.size()[0], 1, 1, 1)
#         print(input_states.size())
        next_states = self.ae_model.predict_batch(input_states, actions).clamp(min = 0, max = 1)
        return next_states, actions
        
    def predict(self, current_state):
        if self.last_action is None:
            self.last_action = 0
        self.current_state = current_state.copy()
        with torch.no_grad():
            current_state = FloatTensor(current_state)
            best_q_value, best_action, root = self.tree_search(current_state, self.last_action)
#         print(best_action, best_q_value)
        self.last_action = best_action
        self.root = root
        return best_action, best_q_value
    
    def show_future(self, key = None):
        if self.root is None:
            print("no explanation")
            return 
        print("Foreseeable action:")
        if key is None:
            best_child = self.root.best_child
            title = "if " +  ACTION_DICT_REVERSE[self.root.best_action]
        else:
            best_child = self.root.children[key]
            title = "if " + ACTION_DICT_REVERSE[key + 1]
        cv2.imshow(title, self.current_state.swapaxes(0, 2).swapaxes(0, 1))
        print("If the agent move: {}".format(title))
        cv2.waitKey(1000)
        count = 1
        while best_child is not None:
            obs = best_child.state.swapaxes(0, 2).swapaxes(0, 1)
            cv2.imshow(title, obs)
            
            if best_child.best_child is not None:
                print("Foreseeable future {} step: {}".format(count, ACTION_DICT_REVERSE[best_child.best_action]))
            count += 1
#             print(obs)
            cv2.waitKey(1000)
            best_child = best_child.best_child
        
#         print()
