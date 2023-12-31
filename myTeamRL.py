# baselineTeam.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import util

from captureAgents import CaptureAgent
from game import Directions
from util import nearestPoint
from environment import PacmanEnv
from capture import AgentRules
from game import Actions
ACTION_NAMES = ['North', 'South', 'West', 'East', 'Stop']

NUM_AGENTS = 4


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='ReinforcementLearningAgent', second='ReinforcementLearningAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]

##########
# Agents #
##########
class ReinforcementLearningAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        # This is set by the testing script. When deploying, this needs to be replaced with a load from a file
        self.rl = None
        self.start = None

    def register_initial_state(self, game_state):
        self.is_red = self.index in game_state.red_team 
        if self.is_red:
            self.friendlies = game_state.red_team
            self.enemies = game_state.blue_team
            self.enemy_positions = [game_state.get_agent_position(enemy) for enemy in self.enemies]
        else:
            self.friendlies = game_state.blue_team
            self.enemies = game_state.red_team
            self.enemy_positions = [game_state.get_agent_position(enemy) for enemy in self.enemies]
        self.height = game_state.data.layout.height
        self.width = game_state.data.layout.width
        self.prev_act = None
        # TODO: use the start positions of the enemies somehow
        self.enemy_positions = [game_state.get_agent_position(self.enemies[0]), game_state.get_agent_position(self.enemies[1])]
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        state_dict = PacmanEnv.get_state_dict(self, game_state, self.index, self.friendlies, self.enemies, self.height, self.width)
        # state_dict['img'] = state_dict['img'].numpy()
        pred = self.rl.predict(state_dict)
        act = ACTION_NAMES[pred[0]]
        
        if act not in game_state.get_legal_actions(self.index):
            act = random.choice(game_state.get_legal_actions(self.index))
        return act