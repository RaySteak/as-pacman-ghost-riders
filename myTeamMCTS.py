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
import numpy as np
import time
from copy import deepcopy

from captureAgents import CaptureAgent
from game import Directions
from util import nearestPoint
from environment import PacmanEnv
from capture import AgentRules
from game import Configuration
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


def my_legal_actions(state, agent_index):
    return [act for act in state.get_legal_actions(agent_index) if act != Directions.STOP]

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

enemy_positions = None

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
        global enemy_positions
        if self.red:
            self.friendlies = game_state.red_team
            self.enemies = game_state.blue_team
        else:
            self.friendlies = game_state.blue_team
            self.enemies = game_state.red_team
        self.height = game_state.data.layout.height
        self.width = game_state.data.layout.width
        self.prev_act = None
        # TODO: use the start positions of the enemies somehow
        enemy_positions = [game_state.get_agent_position(self.enemies[0]), game_state.get_agent_position(self.enemies[1])]
        CaptureAgent.register_initial_state(self, game_state)

    def is_fully_expanded(self, node):
        return len(node.children) >= len(my_legal_actions(node.state, node.agent_index))
        

    def get_not_fully_expanded(self, node):
        if self.is_fully_expanded(node):
            if node.agent_index in self.enemies:
                sorted_children = sorted(node.children, key=lambda x: x.value, reverse=False)
            else:
                sorted_children = sorted(node.children, key=lambda x: x.value, reverse=True)
            return self.get_not_fully_expanded(sorted_children[0])
        
        for action in my_legal_actions(node.state, node.agent_index):
            if action in node.visited_actions:
                continue
            successor = node.state.generate_successor(node.agent_index, action)
            node.visited_actions.append(action)
            
            next_agent_index = (node.agent_index + 1) % NUM_AGENTS
            # if successor.data.agent_states[next_agent_index].get_position() is None:
                # next_agent_index = (next_agent_index + 1) % NUM_AGENTS
            
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
        is_red = agent_index in state.red_team
        
        state_score = state.get_score() if is_red else -state.get_score()
        
        friendlies = state.red_team if is_red else state.blue_team
        enemies = state.blue_team if is_red else state.red_team

        friendlies_carrying = [state.data.agent_states[friendlies[0]].num_carrying,
                                 state.data.agent_states[friendlies[1]].num_carrying]
        enemies_carrying = [state.data.agent_states[enemies[0]].num_carrying,
                                 state.data.agent_states[enemies[1]].num_carrying]
        
        food_list = state.get_red_food().as_list() if is_red else state.get_blue_food().as_list()
        my_pos = state.get_agent_state(agent_index).get_position()
        if len(food_list) > 2:
            dist_to_objective = min([self.get_maze_distance(my_pos, food) for food in food_list])
        else:
            # TODO: replace with minimimum maze distance to a border cell on the friendly side
            friendly_side_center = (self.width / 2 + self.width / 4 * (-1 if self.red else 1), self.height / 2)
            dist_to_objective = util.manhattanDistance(my_pos, friendly_side_center)
        
        return 100 * state_score \
               + friendlies_carrying[0] + friendlies_carrying[1] \
               - enemies_carrying[0] - enemies_carrying[1] \
               - 0.1 * dist_to_objective
    
    def rollout_policy(self, state, agent_index, prev_action):
        epsilon = 0.2
        prev_action_weight = 0.1
        
        legal_actions = my_legal_actions(state, agent_index)
        # act_values = []
        # for act in legal_actions:
        #     dir = Actions.direction_to_vector(act)
        #     old_pos = state.data.agent_states[agent_index].get_position()
        #     new_pos = (old_pos[0] + dir[0], old_pos[1] + dir[1])
            
        #     food_list = self.get_food(state).as_list()
        #     food_dist_list = [self.get_maze_distance(new_pos, food) for food in food_list]
        #     if len(food_dist_list) <= 2:
        #         friendly_side_center = (self.width / 2 + self.width / 4 * (-1 if self.red else 1), self.height / 2)
        #         val = util.manhattanDistance(new_pos, friendly_side_center)
        #     else:
        #         val = min(food_dist_list)
        #     act_values.append(val)
        # best_act = sorted([(act, val) for act, val in zip(legal_actions, act_values)], key=lambda x : x[1])[0][0]
        
        # if random.uniform(0, 1) < epsilon:
        p = np.array([1 / len(legal_actions)] * len(legal_actions))
        for i, act in enumerate(legal_actions):
            if act == prev_action:
                p[i] *= prev_action_weight
        p /= np.sum(p)
        best_act = np.random.choice(legal_actions, replace = False, p = p)
        
        return best_act
    
    def rollout(self, node):
        max_depth = 20
        gamma = 0.9
        
        visited = set()
        state = node.state.deep_copy()
        
        visited.add(state.__hash__())
        agent_index = node.agent_index
        
        rollout_value = 0
        depth = 0
        prev_actions = [None] * NUM_AGENTS
        while state.data.timeleft > 0 and (not state.is_over()) and depth < max_depth:
            best_act = self.rollout_policy(state, agent_index, prev_actions[agent_index])
                
            prev_actions[agent_index] = best_act
            
            # state = state.generate_successor(agent_index, best_act)
            state = self.apply_action(state, agent_index, best_act)
            
            state_value = self.evaluate_state(agent_index, state)
            if agent_index in self.enemies:
                state_value = -state_value 
            
            rollout_value += (gamma ** (depth // 2)) * state_value
            
            # Skip over agents whose positions we don't know
            agent_index = (agent_index + 1) % NUM_AGENTS
            if state.data.agent_states[agent_index].get_position() is None:
                print("WADAFAK")
                state.data.timeleft -= 1
                agent_index = (agent_index + 1) % NUM_AGENTS
            
            depth += 1
        
        return rollout_value
     
    def backpropagate(self, node, reward):
        while node is not None:
            node.update(reward)
            node = node.parent

    def mcts(self, game_state):
        # Put estimated positions of enemies into the game state
        for i, pos in enumerate(enemy_positions):
            game_state.data.agent_states[self.enemies[i]].configuration = Configuration(pos, Directions.STOP)
        
        entry_time = time.perf_counter()
        self.in_tree = set()
        self.in_tree.add(game_state.__hash__())
        root = TreeNode(game_state, self.index)
        while True:
            # print(f'For {i}, reward is {root.value}')
            t = time.perf_counter()
            unvisited_node = self.get_not_fully_expanded(root)
            reward = self.rollout(unvisited_node)
            self.backpropagate(unvisited_node, reward)
            if time.perf_counter() - entry_time > 1 - 0.05:
                print('Time taken: ', time.perf_counter() - entry_time)
                print('Iterations: ', root.visits)
                break
        
        return root
    
    def select_best_mcts_child(self, root):
        best_node = sorted(root.children, key=lambda x: x.value, reverse=True)[0]
        return best_node
    
    def select_worst_mcts_child(selg, root):
        worst_node = sorted(root.children, key=lambda x: x.value, reverse=False)[0]
        return worst_node

    def choose_action(self, game_state):
        print(self.friendlies)
        self.friendlies.reverse()
        print(self.friendlies)
        print("CUR INDEX: ", self.index)
        # TODO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! WHY IS "AT START" DIFFERENT FROM "AFTER"
        print("AT START: ", enemy_positions)
        # Set estimated enemy positions to atual positions when we can see them
        for i in range(len(game_state.data.agent_states)):
            if i in self.enemies and game_state.data.agent_states[i].get_position() is not None:
                enemy_positions[i] = game_state.data.agent_states[i].get_position()
        # Run MCTS
        mcts_root = self.mcts(game_state)
        # Choose best (for us) action for our agent
        best_child = self.select_best_mcts_child(mcts_root)
        act = best_child.action
        # Choose worst (for us) action for our enemy
        best_enemy_child = self.select_worst_mcts_child(best_child)
        enemy_act = best_enemy_child.action
        
        state_after_enemy_act = self.apply_action(best_child.state, best_child.agent_index, enemy_act)
        print("Before: ", enemy_positions)
        for i in range(len(enemy_positions)):
            enemy_positions[i] = deepcopy(state_after_enemy_act.get_agent_position(self.enemies[i]))
        print("After: ", enemy_positions)
        # print(self.enemy_positions)
        return act