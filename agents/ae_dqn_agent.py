import torch
import os
import numpy as np

from tqdm import tqdm
from random import uniform, randint
import math
import cv2
from tensorboardX import SummaryWriter

from models.dqn_model import DQNModel
from models.auto_encoder import AEModel
from memory.memory import ReplayBuffer

from utils import clear_summary_path
ACTION_DICT = {
    "STOP": 0,
    "FORWARD": 1,
    "LEFT": 2,
    "RIGHT":3
}
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
class AE_DQN_agent(object):
    def __init__(self, env, hyper_params, action_space = len(ACTION_DICT)):
        
        self.env = env
        self.max_episode_steps = hyper_params['max_steps']
        
        self.beta = hyper_params['beta']
        self.initial_epsilon = 1
        self.final_epsilon = hyper_params['final_epsilon']
        self.epsilon_decay_steps = hyper_params['epsilon_decay_steps']

        self.episode = 0
        self.steps = 0
        self.best_reward = -100000
        self.learning = True
        self.action_space = action_space

        state = env.reset()
        input_len = len(state['depth'])
        output_len = action_space
        self.eval_model = DQNModel(input_len, output_len, learning_rate = hyper_params['learning_rate'])
        self.use_target_model = hyper_params['use_target_model']
        
        if self.use_target_model:
            self.target_model = DQNModel(input_len, output_len)
#         memory: Store and sample experience replay.
        self.memory = ReplayBuffer(hyper_params['memory_size'])
        
        self.batch_size = hyper_params['batch_size']
        self.update_steps = hyper_params['update_steps']
        self.model_replace_freq = hyper_params['model_replace_freq']
        
        # x: 1.8805575370788574, y: 12.196488380432129
        self.goal = [2, 12]
        self.last_pos = [0, 0]
        
        clear_summary_path('dqn_agent_summary/')
        self.summary = SummaryWriter(log_dir = 'dqn_agent_summary/')
        
        self.ae_model = AEModel(1, 2048, learning_rate = hyper_params['learning_rate'])
        self.AE_train_epoch = hyper_params['AE_train_epoch']
    
    def eval_mode(self):
        self.ae_model.aemodel.eval()
        self.eval_model.model.eval()
        self.target_model.model.eval()
    # Linear decrease function for epsilon
    def linear_decrease(self, initial_value, final_value, curr_steps, final_decay_steps):
        decay_rate = curr_steps / final_decay_steps
        if decay_rate > 1:
            decay_rate = 1
        return initial_value - (initial_value - final_value) * decay_rate
    
    def explore_or_exploit_policy(self, state, position):
        p = uniform(0, 1)
        # Get decreased epsilon
        epsilon = self.linear_decrease(self.initial_epsilon, 
                               self.final_epsilon,
                               self.steps,
                               self.epsilon_decay_steps)

        if p < epsilon:
            #return action, None
            new_pos_x, new_pos_y = position[0], position[2]
            goal_x, goal_y = self.goal[0], self.goal[1]
            new_dist = math.sqrt((new_pos_x - goal_x) ** 2 + (new_pos_y - goal_y) ** 2)
            
            if new_dist < 2:
                action = np.random.choice(self.action_space, 1, p=[0.7, 0.1, 0.1, 0.1])[0]
#             print(action)
            else:
                action = np.random.choice(self.action_space, 1, p=[0.1, 0.5, 0.2, 0.2])[0]
            return action
        else:
            #return action, Q-value
            return self.greedy_policy(state)
    def random_policy_ae(self):
        return np.random.choice(self.action_space, 1, p=[0.001, 0.5, 0.25, 0.249])[0]
    
    def greedy_policy(self, state):
        return self.eval_model.predict(state)[0]
    
    def update_batch(self):
#         print(self.update_steps)
        if len(self.memory) < self.batch_size or self.steps % self.update_steps != 0:
            return

        batch = self.memory.sample(self.batch_size)

        (states, actions, reward, next_states,
         is_terminal) = batch
        
        states = states
        next_states = next_states
        terminal = FloatTensor([1 if t else 0 for t in is_terminal])
        reward = FloatTensor(reward)
        batch_index = torch.arange(self.batch_size,
                                   dtype=torch.long)
        
        # Current Q Values
        q_values = self.eval_model.predict_batch(states)
        q_values = q_values[batch_index, actions]
        
        # Calculate target
        if self.use_target_model:
            q_next = self.target_model.predict_batch(next_states)
        else:
            q_next = self.eval_model.predict_batch(next_states)
            
        
        q_max, _ = torch.max(q_next, dim = 1)
        q_max = (1 - terminal) * q_max
        q_target = reward + self.beta * q_max
        
#         print(reward)
        # update model
#         start = time.time()
        loss = self.eval_model.fit(q_values, q_target)
        self.summary.add_scalar("loss", float(loss), self.steps)
#         print(time.time() - start)
    
    def update_batch_ae(self):
        if len(self.memory) < self.batch_size or self.steps % self.update_steps != 0:
            return

        batch = self.memory.sample(self.batch_size)

        (states, actions, _, next_states,
         _) = batch
        
        next_states = FloatTensor(next_states)
        batch_index = torch.arange(self.batch_size,
                                   dtype=torch.long)
        actions = LongTensor(actions)
#         print(actions)
        actions_vector = FloatTensor(np.zeros((self.batch_size, self.action_space)))
        actions_vector[batch_index, actions] = 1
#         print(actions_vector)
        predict_next_state = self.ae_model.predict_batch(states, actions_vector)
#         print(predict_next_state)
#         print(next_states)
#         input()
        loss = self.ae_model.fit(predict_next_state, next_states)
        self.summary.add_scalar("loss_ae", float(loss), self.steps)
        
    def reward_func(self, state, action):
        
        reward = -0.1
        done = False
        new_pos_x, new_pos_y = state['agent_position'][0], state['agent_position'][2]
        last_pos_x, last_pos_y = self.last_pos[0], self.last_pos[1]
        goal_x, goal_y = self.goal[0], self.goal[1]
#         print(self.last_pos)
#         print(new_pos)
        last_dist = math.sqrt((last_pos_x - goal_x) ** 2 + (last_pos_y - goal_y) ** 2)
        new_dist = math.sqrt((new_pos_x - goal_x) ** 2 + (new_pos_y - goal_y) ** 2)
        reward += 10 * (last_dist - new_dist)
        
#         print(reward)
        self.last_pos = [new_pos_x, new_pos_y]
        if action == 0:
            done = True
            reward = 100 / new_dist
            
            if new_dist < 2:
                reward = 100
#         if new_dist < 2 and done:
#             reward = 100
            
        return reward, done
    def learn_and_evaluate(self, training_episodes, test_interval):
        test_number = training_episodes // test_interval
        all_results = []
        
        for i in range(test_number):
            # learn
            self.learn_ae()
            print(self.steps)
            self.steps = 0
            self.learn_all(test_interval)
            print(self.steps)
            # evaluate
            avg_reward = self.evaluate(i)
            all_results.append(avg_reward)
            
        return all_results
    
    def learn_ae(self):
        for episode in tqdm(range(self.AE_train_epoch), desc="Training AE"):
            if (episode + 1) % 200 == 0:
                self.save_ae_model()
            state = self.env.reset()
            done = False
            steps = 0
            self.last_pos = [state['agent_position'][0], state['agent_position'][2]]
            while steps < self.max_episode_steps and not done:
                steps += 1
                self.steps += 1
                
                state_depth = state['depth'].swapaxes(0, 2).swapaxes(1, 2).copy()
                action = self.random_policy_ae()
                next_state = self.env.step(action)
                reward, done = self.reward_func(next_state, action)
                next_state_depth = next_state['depth'].swapaxes(0, 2).swapaxes(1, 2).copy()
                
                if steps < self.max_episode_steps and not done:
                    self.memory.add(state_depth.copy(), action, reward, next_state_depth.copy(), 0)
                else:
                    self.memory.add(state_depth.copy(), action, reward, next_state_depth.copy(), 1)
               
                self.update_batch_ae()
                state = next_state
                
    def learn_all(self, test_interval):
        for episode in tqdm(range(test_interval), desc="Training"):
            state = self.env.reset()
            done = False
            steps = 0
            self.last_pos = [state['agent_position'][0], state['agent_position'][2]]
#             all_rewards = []
            while steps < self.max_episode_steps and not done:
                steps += 1
                self.steps += 1
                
                state_depth = state['depth'].swapaxes(0, 2).swapaxes(1, 2).copy()
#                 print(state_depth.shape)
#                 input()
                action = self.explore_or_exploit_policy(state_depth, state['agent_position'])
                next_state = self.env.step(action)
#                 print(action)
#                 print(next_state["depth"])
#                 cv2.imshow("Image", next_state["depth"])
                
                reward, done = self.reward_func(next_state, action)
#                 all_rewards.append(reward)
#                 input()
                next_state_depth = next_state['depth'].swapaxes(0, 2).swapaxes(1, 2).copy()
                if steps < self.max_episode_steps and not done:
                    self.memory.add(state_depth.copy(), action, reward, next_state_depth.copy(), 0)
                else:
                    self.memory.add(state_depth.copy(), action, reward, next_state_depth.copy(), 1)
               
                self.update_batch()
#                 print(time.time() - start)
                if self.steps % self.model_replace_freq == 0 and self.use_target_model:
                    self.target_model.replace(self.eval_model)
                state = next_state
#             print(all_rewards)
    def evaluate(self, num, trials = 10):
        total_reward = 0
        total_success = 0
        for _ in tqdm(range(trials), desc="Evaluating"):
            state = self.env.reset()
            done = False
            steps = 0
            self.last_pos = [state['agent_position'][0], state['agent_position'][2]]
#             count
            last_action = -1
            while steps < self.max_episode_steps and not done:
                steps += 1
                state_depth = state['depth'].swapaxes(0, 2).swapaxes(1, 2).copy()
                action = self.greedy_policy(state_depth.copy())
#                 print(action)
#                 input()
                if action == 2 and last_action == 3:
                    total_reward += -50
                    break
                if action == 3 and last_action == 2:
                    total_reward += -50
                    break
                state = self.env.step(action)
                reward, done = self.reward_func(state, action)
                if reward == 100:
                    total_success += 1
                total_reward += reward
                last_action = action
#                 cv2.imshow("RGB", next_state["depth"])
        avg_reward = total_reward / trials
        avg_sucess = total_success / trials
        self.summary.add_scalar("reward", avg_reward, num)
        self.summary.add_scalar("success_rate", avg_sucess, num)
        print(avg_reward, avg_sucess)
        f = open('result_file', "a+")
        f.write(str(avg_reward) + "\n")
        f.write(str(avg_sucess) + "\n")
        f.close()
        if avg_reward >= self.best_reward:
            self.best_reward = avg_reward
            self.save_model_best()
        self.save_model()
        return avg_reward

    # save model
    def save_model_best(self):
        self.eval_model.save('save_models/best_model.pt')
        
    def save_model(self):
        self.eval_model.save('save_models/temp_model.pt') 
        
    def save_ae_model(self):
        self.ae_model.save('save_models/ae_model.pt') 
    # load model
    def load_model_best(self):
        self.eval_model.load('save_models/best_model.pt')
        
    def load_model(self):
        self.eval_model.load('save_models/temp_model.pt')
        
    def load_ae_model(self):
        self.ae_model.load('save_models/ae_model.pt') 
        print("loaded ae model")
