import habitat
import cv2
from config_custom import config_custom
from agents.dqn_agent import DQN_agent
from agents.ae_dqn_agent import AE_DQN_agent

hyperparams = {
    'epsilon_decay_steps' : 500000, 
    'final_epsilon' : 0.05,
    'batch_size' : 32, 
    'update_steps' : 10, 
    'memory_size' : 10000, 
    'beta' : 0.99, 
    'model_replace_freq' : 300,
    'learning_rate' : 0.0001,
    'use_target_model': True,
    'max_steps' : 500,
    'AE_train_epoch' : 100000
}
ACTION_DICT = {
    "STOP": 0,
    "FORWARD": 1,
    "LEFT": 2,
    "RIGHT":3
}
actions = [0, 1, 2, 3] # Stop, Forward, Left, Right
def main():
    config = config_custom(path = "configs/tasks/pointnav_my.yaml")
    env = habitat.Env(
        config=config
    )
#     print(env.observation_space)
    print("Environment creation successful")
#     observations = env.reset()
    
#     training_episodes, test_interval = 1000000, 2000
#     agent = DQN_agent(env, hyperparams)
#     result = agent.learn_and_evaluate(training_episodes, test_interval)
    
    training_episodes, test_interval = 1000000, 2000
    
    ae_dqn_agent = AE_DQN_agent(env, hyperparams)
    ae_dqn_agent.load_model_best_dqn()
    ae_dqn_agent.load_model_best_ae()
    result = ae_dqn_agent.learn_and_evaluate(training_episodes, test_interval)
    
if __name__ == "__main__":
    main()