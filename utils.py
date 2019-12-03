import os
import logging
import shutil
from openpyxl import Workbook

def clear_summary_path(path_to_summary):
    """ Removes the summaries if it exists """
    if os.path.exists(path_to_summary):
        logging.info("Summaries Exists. Deleting the summaries at %s" % path_to_summary)
        shutil.rmtree(path_to_summary)
        
class Node():
    def __init__(self, name, state, reward = 0, parent = None, parent_action = None, best_q_value = None, action_name = None):
        
        self.parent = parent
        self.best_q_value = best_q_value
        self.action_name = action_name
        
        self.children = []
        self.action_dict = {}
        self.actions = []
        self.state = state
        self.name = name
        self.best_child = None
        self.best_action = None
        self.parent_action = parent_action
        
    def add_child(self, sub_node, action = None):
        self.children.append(sub_node)
        self.action_dict[str(action)] = sub_node