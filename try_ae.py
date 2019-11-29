import habitat
import cv2
from typing import Any
from agents.ae_dqn_agent import AE_DQN_agent

import numpy as np
from gym import spaces
import torch

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor

FORWARD_KEY="w"
LEFT_KEY="a"
RIGHT_KEY="d"
FINISH="f"

VIEW_FORWARD_KEY="u"
VIEW_LEFT_KEY="h"
VIEW_RIGHT_KEY="k"
VIEW_FINISH="j"


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
    'AE_train_epoch' : 10000
}
# Define the measure and register it with habitat
# By default, the things are registered with the class name
@habitat.registry.register_measure
class EpisodeInfoExample(habitat.Measure):
    def __init__(self, sim, config, **kwargs: Any):
        # This measure only needs the config
        self._config = config

        super().__init__()

    # Defines the name of the measure in the measurements dictionary
    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "episode_info"

    # This is called whenver the environment is reset
    def reset_metric(self, *args: Any, episode, **kwargs: Any):
        # Our measure always contains all the attributes of the episode
        self._metric = vars(episode).copy()
        # But only on reset, it has an additional field of my_value
        self._metric["my_value"] = self._config.VALUE

    # This is called whenver an action is taken in the environment
    def update_metric(self, *args: Any, episode, action, **kwargs: Any):
        # Now the measure will just have all the attributes of the episode
        self._metric = vars(episode).copy()


# Define the sensor and register it with habitat
# For the sensor, we will register it with a custom name
@habitat.registry.register_sensor(name="my_supercool_sensor")
class AgentPositionSensor(habitat.Sensor):
    def __init__(self, sim, config, **kwargs: Any):
        super().__init__(config=config)

        self._sim = sim
        # Prints out the answer to life on init
        print("The answer to life is", self.config.ANSWER_TO_LIFE)

    # Defines the name of the sensor in the sensor suite dictionary
    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "agent_position"

    # Defines the type of the sensor
    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return habitat.SensorTypes.POSITION

    # Defines the size and range of the observations of the sensor
    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(3,),
            dtype=np.float32,
        )

    # This is called whenver reset is called or an action is taken
    def get_observation(
        self, observations, *args: Any, episode, **kwargs: Any
    ):
        return self._sim.get_agent_state().position



def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

def example():
    config = habitat.get_config("configs/tasks/pointnav_my.yaml")
    config.defrost()

    # Add things to the config to for the measure
    config.TASK.EPISODE_INFO_EXAMPLE = habitat.Config()
    # The type field is used to look-up the measure in the registry.
    # By default, the things are registered with the class name
    config.TASK.EPISODE_INFO_EXAMPLE.TYPE = "EpisodeInfoExample"
    config.TASK.EPISODE_INFO_EXAMPLE.VALUE = 5
    # Add the measure to the list of measures in use
    config.TASK.MEASUREMENTS.append("EPISODE_INFO_EXAMPLE")

    # Now define the config for the sensor
    config.TASK.AGENT_POSITION_SENSOR = habitat.Config()
    # Use the custom name
    config.TASK.AGENT_POSITION_SENSOR.TYPE = "my_supercool_sensor"
    config.TASK.AGENT_POSITION_SENSOR.ANSWER_TO_LIFE = 42
    # Add the sensor to the list of sensors in use
    config.TASK.SENSORS.append("AGENT_POSITION_SENSOR")
    config.freeze()
    
    env = habitat.Env(
        config=config
    )
    with torch.no_grad():
        agent = AE_DQN_agent(env, hyperparams)
        agent.load_ae_model()
        agent.eval_mode()
#     agent.load_model()
    print("Environment creation successful")
    observations = env.reset()
#     print(observations)
    print("Destination, distance: {:3f}, theta(radians): {:.2f}".format(
        observations["pointgoal_with_gps_compass"][0], observations["pointgoal_with_gps_compass"][1]))
    cv2.imshow("", observations["depth"])#transform_rgb_bgr(observations["rgb"]))
    
    print("Agent stepping around inside environment.")
    print(env.observation_space)
    count_steps = 0
    view_flag = False
    while not env.episode_over:
        keystroke = cv2.waitKey(0)
        
        if keystroke == ord(FORWARD_KEY):
            action = 1#habitat.SimulatorActions.MOVE_FORWARD
            print("action: FORWARD")
            view_flag = False
        elif keystroke == ord(LEFT_KEY):
            action = 2#habitat.SimulatorActions.TURN_LEFT
            print("action: LEFT")
            view_flag = False
        elif keystroke == ord(RIGHT_KEY):
            action = 3#habitat.SimulatorActions.TURN_RIGHT
            print("action: RIGHT")
            view_flag = False
        elif keystroke == ord(FINISH):
            action = 0#habitat.SimulatorActions.STOP
            print("action: FINISH")
            view_flag = False
#         elif keystroke == ord(AGENT):
#             state_depth = observations['depth'].swapaxes(0, 2).swapaxes(1, 2)
# #             action = agent.greedy_policy(state_depth)
#             action_vector = 
#             action, q = agent.eval_model.predict(state_depth)
#             print(q)
        elif keystroke == ord(VIEW_FORWARD_KEY):
            action = np.array([0, 1, 0, 0])
            if view_flag:
                state_depth = view_obs.swapaxes(0, 2).swapaxes(1, 2)
            else:
                state_depth = observations['depth'].swapaxes(0, 2).swapaxes(1, 2)
                view_flag = True 
        elif keystroke == ord(VIEW_LEFT_KEY):
            action = np.array([0, 0, 1, 0])
            if view_flag:
                state_depth = view_obs.swapaxes(0, 2).swapaxes(1, 2)
            else:
                state_depth = observations['depth'].swapaxes(0, 2).swapaxes(1, 2)
                view_flag = True 
        elif keystroke == ord(VIEW_RIGHT_KEY):
            action = np.array([0, 0, 0, 1])
            if view_flag:
                state_depth = view_obs.swapaxes(0, 2).swapaxes(1, 2)
            else:
                state_depth = observations['depth'].swapaxes(0, 2).swapaxes(1, 2)
                view_flag = True 
        elif keystroke == ord(VIEW_FINISH):
            action = np.array([1, 0, 0, 0])
            if view_flag:
                state_depth = view_obs.swapaxes(0, 2).swapaxes(1, 2)
            else:
                state_depth = observations['depth'].swapaxes(0, 2).swapaxes(1, 2)
                view_flag = True 
        else:
            print("INVALID KEY")
            continue
        
        if view_flag:
            with torch.no_grad():
                
#                 print(state_depth)
#                 print(state_depth.shape)
#                 state_depth = observations['depth'].swapaxes(0, 2).swapaxes(1, 2)
                action = FloatTensor(action)
                view_obs = agent.ae_model.predict(state_depth, action)[0]
                view_obs = view_obs.detach().cpu().numpy().swapaxes(0, 2).swapaxes(0, 1)
#                 print(view_obs.shape)
                
                view_obs = np.clip(view_obs, 0, 1)
#                 print(view_obs)
            cv2.imshow("view", view_obs)
            continue
            
        observations = env.step(action)
#         reward, done = agent.reward_func(observations, action)
#         print(reward, done)
        count_steps += 1
#         env = habitat.Env(
#             config=config
#         )
#         observations = env.reset()
# x:1.9, y: 0.17
#         print("Destination, distance: {:3f}, theta(radians): {:.2f}".format(
#             observations["pointgoal_with_gps_compass"][0], observations["pointgoal_with_gps_compass"][1]))
        print("x: {}, y: {}".format(
            observations["agent_position"][0], observations["agent_position"][2]))
        cv2.imshow("", observations["depth"])#transform_rgb_bgr(observations["rgb"]))

    print("Episode finished after {} steps.".format(count_steps))

    if action == 0 and observations["pointgoal_with_gps_compass"][0] < 0.2:
        print("you successfully navigated to destination point")
    else:
        print("your navigation was unsuccessful")


if __name__ == "__main__":
    example()