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
from capture import AgentRules, SIGHT_RANGE
from game import Configuration, Directions, Actions

NUM_AGENTS = 4
# Hyperparameters
max_depth = 10 # Max depth to which to do rollouts
discard_enemy_pos_threshold = 9999999999 # After how many moves of not seeing the enemy to discard the estimated enemy position
gamma = 0.9 # Discount factor for rollouts
epsilon = 0.1 # Probability of choosing random action in the rollout policy
prev_action_weight = 0.001 # Weight of undoing previous action in the rollout policy when choosing a random action (the lower, the less likely the random action will undo the previous action)
sigmoid_cutoff = 6 # The values of epsilon2 will be taken by sampling the sigmoid from [-sigmoid_cutoff, sigmoid_cutoff]

#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='DefensiveMCTSAgent', second='OffensiveMCTSAgent', num_training=0):
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
not_seen_for = [0, 0]
prev_friendly_food_list = None

class MCTSAgent(CaptureAgent):
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.total_dist_to_invading_enemies_weight = None

    def register_initial_state(self, game_state):
        global enemy_positions
        CaptureAgent.register_initial_state(self, game_state)
        
        if self.red:
            self.friendlies = game_state.red_team
            self.enemies = game_state.blue_team
        else:
            self.friendlies = game_state.blue_team
            self.enemies = game_state.red_team
        self.height = game_state.data.layout.height
        self.width = game_state.data.layout.width
        enemy_positions = [game_state.get_agent_position(self.enemies[0]), game_state.get_agent_position(self.enemies[1])]

    def is_fully_expanded(self, node):
        return len(node.children) >= len(my_legal_actions(node.state, node.agent_index))
    
    def get_action_probabilities(self, legal_actions, prev_action):
        p = np.array([1 / len(legal_actions)] * len(legal_actions))
        for i, act in enumerate(legal_actions):
            if Directions.REVERSE[act] == prev_action:
                p[i] *= prev_action_weight
        p /= np.sum(p)
        return p

    def get_not_fully_expanded(self, node):
        if self.is_fully_expanded(node):
            if node.agent_index in self.enemies:
                sorted_children = sorted(node.children, key=lambda x: x.value, reverse=False)
            else:
                sorted_children = sorted(node.children, key=lambda x: x.value, reverse=True)
            
            for child in sorted_children:
                ret = self.get_not_fully_expanded(child)
                if ret is not None:
                    return ret
        
        legal_actions = my_legal_actions(node.state, node.agent_index)
        action_probabilities = self.get_action_probabilities(legal_actions, node.action)
        actions_prob_list = [(act, prob) for act, prob in zip(legal_actions, action_probabilities)]
        actions_prob_list = sorted(actions_prob_list, key=lambda x: x[1], reverse=True)
        legal_actions = [act for act, _ in actions_prob_list]
        for action in my_legal_actions(node.state, node.agent_index):
            if action in node.visited_actions:
                continue
            successor = node.state.generate_successor(node.agent_index, action)
            if successor.__hash__() in self.in_tree:
                continue
            node.visited_actions.append(action)
            
            next_agent_index = (node.agent_index + 1) % NUM_AGENTS
            if successor.data.agent_states[next_agent_index].get_position() is None:
                next_agent_index = (next_agent_index + 1) % NUM_AGENTS
            
            node.add_child(TreeNode(successor, next_agent_index, node, action))
            self.in_tree.add(successor.__hash__())
            return node.children[-1]
        return None
    
    def apply_action(self, state, agent_index, action):
        AgentRules.apply_action(state, action, agent_index)
        AgentRules.check_death(state, agent_index)
        AgentRules.decrement_timer(state.data.agent_states[agent_index])

        # Bookkeeping
        state.data._agent_moved = agent_index
        state.data.score += state.data.score_change
        state.data.timeleft = state.data.timeleft - 1
        return state
    
    def dist_to_friendly_side(self, state, my_pos, is_red):
        if is_red and my_pos[0] <= self.width / 2:
            return 0
        if not is_red and my_pos[0] > self.width / 2:
            return 0
        border = [(int(self.width / 2 + (0 if is_red else 1)), i) for i in range(self.height)]
        border_dists = []
        for border_cell in border:
            # Check if cell is wall:
            if not state.has_wall(border_cell[0], border_cell[1]):
                border_dists.append(self.get_maze_distance(my_pos, border_cell))
        return min(border_dists)

    def dist_to_enemy_side(self, state, my_pos, is_red):
        if is_red and my_pos[0] > self.width / 2:
            return 0
        if not is_red and my_pos[0] <= self.width / 2:
            return 0
        border = [(int(self.width / 2 + (1 if self.red else 0)), i) for i in range(self.height)]
        border_dists = []
        for border_cell in border:
            # Check if cell is wall:
            if not state.has_wall(border_cell[0], border_cell[1]):
                border_dists.append(self.get_maze_distance(my_pos, border_cell))
        return min(border_dists)
    
    def evaluate_state(self, agent_index, state):
        is_red = agent_index in state.red_team
        
        state_score = state.get_score() if is_red else -state.get_score()
        
        friendlies = state.red_team if is_red else state.blue_team
        enemies = state.blue_team if is_red else state.red_team

        friendlies_carrying = [state.data.agent_states[friendlies[0]].num_carrying,
                                 state.data.agent_states[friendlies[1]].num_carrying]
        enemies_carrying = [state.data.agent_states[enemies[0]].num_carrying,
                                 state.data.agent_states[enemies[1]].num_carrying]
        
        food_list = state.get_blue_food().as_list() if is_red else state.get_red_food().as_list()
        enemy_food_list = state.get_red_food().as_list() if is_red else state.get_blue_food().as_list()
        my_pos = state.get_agent_state(agent_index).get_position()
        if len(food_list) > 2:
            dist_to_food = min([self.get_maze_distance(my_pos, food) for food in food_list])
        else:
            dist_to_food = self.dist_to_friendly_side(state, my_pos, is_red)
        
        dist_to_friendly_side = self.dist_to_friendly_side(state, my_pos, is_red)
        
        carrying = state.data.agent_states[agent_index].num_carrying
        normalize_carrying = (len(food_list) + carrying)
        normalize_carrying = normalize_carrying if normalize_carrying != 0 else 1
        epsilon2 = 1 / (1 + np.exp(-(2 * carrying / normalize_carrying - 1) * sigmoid_cutoff)) # No food => epsilon2 really small; lots of food => epsilon2 really big
        dist_to_objective = (1 - epsilon2) * dist_to_food + epsilon2 * dist_to_friendly_side
        
        total_dist_to_invading_enemies = 0
        enemy_positions = [state.data.agent_states[enemies[0]].get_position(), state.data.agent_states[enemies[1]].get_position()]
        for enemy_pos in enemy_positions:
            if enemy_pos is None:
                continue
            if (is_red and enemy_pos[0] <= self.width / 2) or ((not is_red) and enemy_pos[0] > self.width / 2):
                total_dist_to_invading_enemies += self.get_maze_distance(my_pos, enemy_pos)
        
        # TODO: penalize how close the enemy food is to the enemy side (shouldn't improve that much)
        # return -5 * dist_to_objective
        food_len_weight = 0
        enemy_food_len_weight = 0
        total_dist_to_invading_enemies_weight = self.total_dist_to_invading_enemies_weight
        if self.total_dist_to_invading_enemies_weight < 1:
            dist_to_friendly_side
            food_len_weight = 30
            dist_to_enemy_side_weight = 20
            # dist_to_enemy_side = self.dist_to_enemy_side(state, my_pos)
        else:
            enemy_food_len_weight = 30
            dist_to_enemy_side_weight = 0
        
        return 100 * state_score \
                + 10 * friendlies_carrying[0] + 10 * friendlies_carrying[1] \
                - 10 * enemies_carrying[0] - 10 * enemies_carrying[1] \
                - total_dist_to_invading_enemies_weight * total_dist_to_invading_enemies \
                - food_len_weight * len(food_list) + enemy_food_len_weight * len(enemy_food_list) \
                - dist_to_enemy_side_weight * self.dist_to_enemy_side(state, my_pos, is_red) \
                - 5 * dist_to_objective
    
    def rollout_policy(self, state, agent_index, prev_action):
        legal_actions = my_legal_actions(state, agent_index)
        # Pick random action
        if np.random.uniform(0, 1) < epsilon:
            p = self.get_action_probabilities(legal_actions, prev_action)
            return np.random.choice(legal_actions, replace = False, p = p)
        is_red = agent_index in state.red_team
        
        food_list = state.get_blue_food().as_list() if is_red else state.get_red_food().as_list()
        carrying = state.data.agent_states[agent_index].num_carrying
        
        eat_food_act_values = []
        score_points_act_values = []
        for act in legal_actions:
            dir = Actions.direction_to_vector(act)
            old_pos = state.data.agent_states[agent_index].get_position()
            new_pos = (old_pos[0] + dir[0], old_pos[1] + dir[1])
            
            food_dist_list = [self.get_maze_distance(new_pos, food) for food in food_list]
            if len(food_dist_list) <= 2:
                eat_food_val = -self.dist_to_friendly_side(state, new_pos, is_red)
            else:
                eat_food_val = -min(food_dist_list)
                
            score_points_val = -self.dist_to_friendly_side(state, new_pos, is_red)
            
            eat_food_act_values.append(eat_food_val)
            score_points_act_values.append(score_points_val)
        
        epsilon2 = 1 / (1 + np.exp(-(2 * carrying / (len(food_list) + carrying) - 1) * sigmoid_cutoff)) # No food => epsilon2 really small; lots of food => epsilon2 really big
        # print(epsilon2)
        
        if np.random.uniform(0, 1) > epsilon2: # When carrying little food, high chance to happen
            return max([(act, val) for act, val in zip(legal_actions, eat_food_act_values)], key=lambda x : x[1])[0]
        return max([(act, val) for act, val in zip(legal_actions, score_points_act_values)], key=lambda x : x[1])[0]
    
    def rollout(self, node):
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
            # print(agent_index)
            if agent_index in self.enemies:
                state_value = -state_value 
            
            rollout_value += (gamma ** (depth)) * (state_value)
            
            agent_index = (agent_index + 1) % NUM_AGENTS
            
            # Skip over agents whose positions we don't know
            # if state.data.agent_states[agent_index].get_position() is None:
            #     state.data.timeleft -= 1
            #     agent_index = (agent_index + 1) % NUM_AGENTS
            if agent_index in self.enemies:
                agent_index = (agent_index + 1) % NUM_AGENTS
            
            depth += 1
        
        return rollout_value
     
    def backpropagate(self, node, reward):
        ascent = 0
        while node is not None:
            node.update(reward)
            node = node.parent
            ascent += 1
    
    def set_pos_on_friendly_side(self, game_state, friendly1_pos, friendly2_pos, i):
        global enemy_positions
        print("SETTING RANDOM POSITION ON FRIENDLY SIDE FOR ENEMY")
        food_list = game_state.get_red_food().as_list() if self.red else game_state.get_blue_food().as_list()
        is_friendly = lambda pos: (pos[0] <= self.width // 2) if self.red else (pos[0] > self.width / 2)

        pos_list = [(x, y) for x in range(self.width) for y in range(self.height) if not game_state.has_wall(x, y) and is_friendly((x, y))]
        pos_array = np.empty(len(pos_list), dtype='object')
        pos_array[:] = pos_list
        probability_list = np.array([(1.0 if ((game_state.has_food(pos[0], pos[1])) and
                                  util.manhattanDistance(pos, friendly1_pos) > SIGHT_RANGE and
                                  util.manhattanDistance(pos, friendly2_pos) > SIGHT_RANGE) else 1.0/100) for pos in pos_list])
        probability_list /= np.sum(probability_list)
        
        rand_pos = np.random.choice(pos_array, p=probability_list)
        enemy_positions[i] = rand_pos
        enemy_positions[i] = (int(enemy_positions[i][0]), int(enemy_positions[i][1]))

        return enemy_positions[i]

    def mcts(self, game_state):
        entry_time = time.process_time()
        # Set enemy positions to estimated positions if they are valid
        # TODO: can an enemy be seen by friendly 1 if friendly 2 is in range only?
        for i, pos in enumerate(enemy_positions):
            if pos is None: # If we get here, we have lost track of the enemy at the previous step (shouldn't happen if threshold is high enough)
                continue
            
            not_seen_by_friendly1 = False
            friendly1_pos = game_state.data.agent_states[self.friendlies[0]].get_position()
            if util.manhattanDistance(friendly1_pos, pos) <= SIGHT_RANGE and game_state.data.agent_states[self.enemies[i]].get_position() is None:
                not_seen_by_friendly1 = True
            not_seen_by_friendly2 = False
            friendly2_pos = game_state.data.agent_states[self.friendlies[1]].get_position()
            if util.manhattanDistance(friendly2_pos, pos) <= SIGHT_RANGE and game_state.data.agent_states[self.enemies[i]].get_position() is None:
                not_seen_by_friendly2 = True
            
            if not_seen_by_friendly1 and not_seen_by_friendly2:
                pos = self.set_pos_on_friendly_side(game_state, friendly1_pos, friendly2_pos, i)
            
            game_state.data.agent_states[self.enemies[i]].configuration = Configuration(pos, Directions.STOP)
        
        self.in_tree = set()
        root = TreeNode(game_state, self.index)
        self.in_tree.add(root.state.__hash__())
        while True:
            # print(f'For {i}, reward is {root.value}')
            t = time.process_time()
            unvisited_node = self.get_not_fully_expanded(root)
            reward = self.rollout(unvisited_node)
            self.backpropagate(unvisited_node, reward)
            if time.process_time() - entry_time > 0.85:
                print('Time taken: ', time.process_time() - entry_time)
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
        global prev_friendly_food_list
        cur_friendly_food_list = game_state.get_red_food().as_list() if self.red else game_state.get_blue_food().as_list()
        
        if prev_friendly_food_list is None:
            missing_food = []
        else:
            missing_food = [pos for pos in prev_friendly_food_list if pos not in cur_friendly_food_list]
        
        prev_friendly_food_list = cur_friendly_food_list
        # Update estimated enemy positions
        for i, enemy_ind in enumerate(self.enemies):
            # Set estimated position to actual position if agent is in sight
            # TODO: also set enemy position by getting the position of the missing food
            if game_state.data.agent_states[enemy_ind].get_position() is not None:
                enemy_pos = game_state.data.agent_states[enemy_ind].get_position()
                enemy_positions[i] = (enemy_pos[0], enemy_pos[1])
                not_seen_for[i] = 0
            # Set estimated position to None if agent hasn't been seen for a long time
            elif len(missing_food) != 0:
                print("FOOD WAS STOLEN")
                enemy_positions[i] = missing_food.pop(0)
                not_seen_for[i] = 0
            else:
                not_seen_for[i] += 1
                if not_seen_for[i] >= discard_enemy_pos_threshold:
                    friendly1_pos = game_state.data.agent_states[self.friendlies[0]].get_position()
                    friendly2_pos = game_state.data.agent_states[self.friendlies[1]].get_position()
                    enemy_positions[i] = self.set_pos_on_friendly_side(game_state, friendly1_pos, friendly2_pos, i)
                    not_seen_for[i] = 0
        # Run MCTS
        mcts_root = self.mcts(game_state)
        # print([(child.value, child.action) for child in mcts_root.children])
        print(f'Enemy positions', enemy_positions)
        # Choose best (for us) action for our agent
        best_child = self.select_best_mcts_child(mcts_root)
        act = best_child.action
        # Choose worst (for us) action for our enemy
        if best_child.agent_index in self.enemies:
            try:
                best_enemy_child = self.select_worst_mcts_child(best_child)
                enemy_act = best_enemy_child.action
            except:
                enemy_act = np.random.choice(my_legal_actions(best_child.state, best_child.agent_index))
        
            state_after_enemy_act = self.apply_action(best_child.state, best_child.agent_index, enemy_act)
            for i in range(len(enemy_positions)):
                enemy_positions[i] = deepcopy(state_after_enemy_act.get_agent_position(self.enemies[i]))
        # print(self.enemy_positions)
        return act

class OffensiveMCTSAgent(MCTSAgent):
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.total_dist_to_invading_enemies_weight = 0.0

class DefensiveMCTSAgent(MCTSAgent):
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.total_dist_to_invading_enemies_weight = 50