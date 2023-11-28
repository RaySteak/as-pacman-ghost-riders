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


def my_legal_actions(state, agent_index, prev_action = None):
    prev_action = None
    legal_actions = []
    for act in state.get_legal_actions(agent_index):
        if act != Directions.STOP:
            legal_actions.append(act)
    # if prev_action is not None and Directions.REVERSE[prev_action] in legal_actions and len(legal_actions) > 1:
        # legal_actions.remove(Directions.REVERSE[prev_action])
    # legal_actions = [act for act in state.get_legal_actions(agent_index) if act != Directions.STOP]
    return legal_actions

##########
# Agents #
##########
class TreeNode:
    def __init__(self, state, agent_index, parent=None, action=None, reward=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.reward = reward
        self.agent_index = agent_index
        self.children = []
        self.visited_actions = []
        self.visits = 0
        self.value = 0

    def add_child(self, child):
        self.children.append(child)

    def update(self, reward):
        self.reward += reward
        self.visits += 1
        self.value = self.reward / self.visits

    def __repr__(self):
        return f"TreeNode({self.state}, {self.parent}, {self.action}, {self.reward}, {self.children}, {self.visits}, {self.value})"

class ReinforcementLearningAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.rl = None
        self.start = None

    def set_rl_model(self, rl):
        self.rl = rl

    def register_initial_state(self, game_state):
        is_red = self.index in game_state.red_team 
        if is_red:
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

    def is_fully_expanded(self, node):
        legal_actions = []
        # print("IN IS FULLY")
        for act in my_legal_actions(node.state, node.agent_index, node.action):
            # succ = node.state.generate_successor(node.agent_index, act)
            # if succ.__hash__() not in self.in_tree:
                # legal_actions.append(act)
            legal_actions.append(act)
        # print(f'IS FULLY')
        # print(len(legal_actions))
        # if len(legal_actions) == 0:
            # return print("IT'S 0")
        return len(node.children) >= len(legal_actions)
        

    def get_not_fully_expanded(self, node):
        if self.is_fully_expanded(node):
            sorted_children = sorted(node.children, key=lambda x: x.value, reverse=True)
            # for child in sorted_children:
            #     ret = self.get_not_fully_expanded(child)
            #     if ret is not None:
            #         return ret
            # print("WHY HEREEEEE1")
            # return None
            return self.get_not_fully_expanded(sorted_children[0])
        
        # print(my_legal_actions(node.state, node.agent_index, node.action))
        for action in my_legal_actions(node.state, node.agent_index, node.action):
            # print(action)
            if action in node.visited_actions:
                continue
            successor = node.state.generate_successor(node.agent_index, action)
            # if successor.__hash__() in self.in_tree:
                # continue
            node.visited_actions.append(action)
            
            next_agent_index = (node.agent_index + 1) % NUM_AGENTS
            if successor.data.agent_states[next_agent_index].get_position() is None:
                next_agent_index = (next_agent_index + 1) % NUM_AGENTS
            
            self.in_tree.add(successor.__hash__())
            node.add_child(TreeNode(successor, next_agent_index, node, action))
            return node.children[-1]
    
    def apply_action(self, state, agent_index, action):
        AgentRules.apply_action(state, action, agent_index)
        AgentRules.check_death(state, agent_index)
        AgentRules.decrement_timer(state.data.agent_states[agent_index])

        # Bookkeeping
        state.data._agent_moved = agent_index
        state.data.score += state.data.score_change
        state.data.timeleft = state.data.timeleft - 1
        return state
    
    def evaluate_state(self, agent_index, state):
        state_score = state.get_score() if self.friendlies == state.red_team else -state.get_score()
        friendlies_carrying = [state.data.agent_states[self.friendlies[0]].num_carrying,
                                 state.data.agent_states[self.friendlies[1]].num_carrying]
        
        food_list = self.get_food(state).as_list()
        my_pos = state.get_agent_state(agent_index).get_position()
        
        if len(food_list) > 2:
            min_dist_to_food = min([self.get_maze_distance(my_pos, food) for food in food_list])
        else:
            min_dist_to_food = 0
        
        return state_score + friendlies_carrying[0] + friendlies_carrying[1] - 0.1 * min_dist_to_food
    
    def rollout(self, node):
        gamma = 0.999
        visited = set()
        state = node.state
        visited.add(state.__hash__())
        agent_index = node.agent_index
        prev_action = None
        
        rollout_value = 0
        depth = 0
        while state.data.timeleft > 0 and not state.is_over():
            #### I
            # # This random policy will be replaced with a neural network for RL
            # unvisited_actions = []
            # for action in my_legal_actions(state, agent_index, prev_action):
            #     child = state.generate_successor(agent_index, action)
            #     if child.__hash__() not in visited:
            #         unvisited_actions.append(action)
            
            # if len(unvisited_actions) == 0:
            #     best_act = random.choice(my_legal_actions(state, agent_index, prev_action))
            # else:
            #     best_act = random.choice(unvisited_actions)
            #### II
            # act_values = []
            # legal_actions = my_legal_actions(state, agent_index, prev_action)
            # for act in legal_actions:
            #     succ = state.generate_successor(agent_index, act)
            #     new_pos = succ.data.agent_states[agent_index].get_position()
            #     dist = min([self.get_maze_distance(new_pos, food) for food in self.get_food(succ).as_list()])
            #     act_values.append(dist)
            #### III
            act_values = []
            legal_actions = my_legal_actions(state, agent_index, prev_action)
            for act in legal_actions:
                dir = Actions.direction_to_vector(act)
                old_pos = state.data.agent_states[agent_index].get_position()
                new_pos = (old_pos[0] + dir[0], old_pos[1] + dir[1])
                
                food_list = self.get_food(state).as_list()
                if len(food_list) > 2:
                    dist = min([self.get_maze_distance(new_pos, food) for food in food_list])
                else:
                    dist = 0
                act_values.append(dist)
            
            best_act = sorted([(act, val) for act, val in zip(legal_actions, act_values)], key=lambda x : x[1])[0][0]
                
            prev_action = best_act
            state = self.apply_action(state, agent_index, best_act)
            visited.add(state.__hash__())
            rollout_value += (gamma ** depth) * self.evaluate_state(agent_index, state)
            depth += 1
            
            # Skip over agents whose positions we don't know
            agent_index = (agent_index + 1) % NUM_AGENTS
            if state.data.agent_states[agent_index].get_position() is None:
                agent_index = (agent_index + 1) % NUM_AGENTS
        
        # Evaluate state score
        return rollout_value
     
    def backpropagate(self, node, reward):
        while node is not None:
            node.update(reward)
            node = node.parent

    def mcts(self, game_state, iterations, prev_act):
        self.in_tree = set()
        self.in_tree.add(game_state.__hash__())
        root = TreeNode(game_state, self.index, None, prev_act)
        for i in range(iterations):
            # print(f'For {i}, reward is {root.value}')
            unvisited_node = self.get_not_fully_expanded(root)
            reward = self.rollout(unvisited_node)
            self.backpropagate(unvisited_node, reward)
        
        return root
    
    def select_best_mcts_action(self, root):
        for child in root.children:
            print(f'For {child.action}, reward is {child.value}')
        best_node = sorted(root.children, key=lambda x: x.value, reverse=True)[0]
        return best_node.action

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        state_dict = PacmanEnv.get_state_dict(game_state, self.index, self.friendlies, self.enemies, self.height, self.width)
        state_dict['img'] = state_dict['img'].numpy()
        pred = self.rl.predict(state_dict)
        act = ACTION_NAMES[pred[0]]
        
        if act not in game_state.get_legal_actions(self.index):
            act = random.choice(game_state.get_legal_actions(self.index))
        return act
        # mcts_root = self.mcts(game_state, 100, None)
        # act = self.select_best_mcts_action(mcts_root)
        # self.prev_act = act
        # return act

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}