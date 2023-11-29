import sys
sys.path.append('../')
import capture
import layout as lay
import gymnasium as gym
import numpy as np
import stable_baselines3 as sb3
from gymnasium import spaces
from torchvision import transforms as T
import textDisplay
import util
from capture import AgentRules
# import myTeamTrain

IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
FEATURES_LENGTH = 6
NUM_AGENTS = 4

ACTION_NAMES = ['North', 'South', 'West', 'East']
NUM_ACTIONS = len(ACTION_NAMES)

TEAM_FILE = 'myteamTrain'

TEAMS = ['RED', 'BLUE']

#TODO: run the algorithm on a random layout each time. Also, look into the layout generation function provided in the codebase
LAYOUT_NAME = "../layouts/defaultCapture.lay"

class PacmanEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, rules, max_episode_length, enemy_team = '../baselineTeam'):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        # The multi-observation must follow this format exactly:
        # https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/envs/multi_input_envs.html#SimpleMultiObsEnv
        self.observation_space = spaces.Dict(
            spaces={
                "vec": spaces.Box(low = 0, high = 1, shape = (FEATURES_LENGTH, ), dtype=np.float64),
                "img": spaces.Box(low = 0, high = 255, shape = (IMAGE_HEIGHT, IMAGE_WIDTH, 1), dtype=np.uint8),
            }
        )
        
        # Initialize
        self.layout = lay.get_layout(LAYOUT_NAME)

        # For training, it doesn't matter what team file we load for the RL agents,
        # we just need to specify a valid team file for it to work. The policy will
        # be provided by the RL algorithm trained on this environment.
        #TODO: run from a random perspective each episode (blue and red switched)
        self.rl_agents = capture.load_agents(False, enemy_team, [])
        self.enemy_agents = capture.load_agents(True, enemy_team, [])

        self.red_agents = self.enemy_agents
        self.blue_agents = self.rl_agents
        
        self.rules = rules
        self.max_episode_length = max_episode_length
    
    @staticmethod    
    def get_state_image(obs_state, agent_index, team_indices, enemy_indices, height, width):
        # TODO: improve image.
        game_obs_str = str(obs_state)
        game_obs = np.copy(np.frombuffer(''.join(game_obs_str.split('\n')[:-2]).encode(), dtype=np.uint8))
        game_obs = np.reshape(game_obs, (height, width))
        
        teammate_index = team_indices[1] if agent_index == team_indices[0] else team_indices[0]
        moving_agent = obs_state.data.agent_states[agent_index]
        # print(agent_index)
        teammate = obs_state.data.agent_states[teammate_index]
        
        # IF they are on top of eachother
        if teammate.is_pacman:
            game_obs[int(teammate.get_position()[1]), int(teammate.get_position()[0])] += ord('T') + ord('P')
        else:
            game_obs[int(teammate.get_position()[1]), int(teammate.get_position()[0])] += ord('T') + ord('G')
        
        if moving_agent.is_pacman:
            game_obs[int(moving_agent.get_position()[1]), int(moving_agent.get_position()[0])] += ord('S') + ord('P')
        else:
            game_obs[int(moving_agent.get_position()[1]), int(moving_agent.get_position()[0])] += ord('S') + ord('G')
        
        enemy1 = obs_state.data.agent_states[enemy_indices[0]]
        enemy2 = obs_state.data.agent_states[enemy_indices[1]]
        if enemy1.get_position() is not None:
            if enemy1.is_pacman:
                game_obs[int(enemy1.get_position()[1]), int(enemy1.get_position()[0])] = ord('E') + ord('P')
            else:
                game_obs[int(enemy1.get_position()[1]), int(enemy1.get_position()[0])] = ord('E') + ord('G')        
        if enemy2.get_position() is not None:
            if enemy2.is_pacman:
                game_obs[int(enemy2.get_position()[1]), int(enemy2.get_position()[0])] = ord('E') + ord('P')
            else:
                game_obs[int(enemy2.get_position()[1]), int(enemy2.get_position()[0])] = ord('E') + ord('G')
        
        game_obs = T.Resize((IMAGE_HEIGHT, IMAGE_WIDTH), T.InterpolationMode.BILINEAR)(T.ToTensor()(game_obs))
        return np.transpose(game_obs, [1, 2, 0])
    
    @staticmethod
    def get_state_vector(obs_state, agent_index, team_indices, enemy_indices, height, width):
        # TODO: use knwoledge of enemy positions (both real and from previous estimates) to filter out the noise (maybe use a Kalman filter or something)
        enemy_dist = np.array([obs_state.agent_distances[enemy_indices[0]], obs_state.agent_distances[enemy_indices[1]]])
        enemy_dist = enemy_dist / (height + width) # Normalize distances
        
        available_dirs = np.zeros(4)
        legal_actions = obs_state.get_legal_actions(agent_index)
        for i in range(4):
            available_dirs[i] = 1 if ACTION_NAMES[i] in legal_actions else 0
        
        return np.concatenate([enemy_dist, available_dirs])
    
    @staticmethod
    def get_state_dict(agent, obs_state, agent_index, team_indices, enemy_indices, height, width):
        game_img = PacmanEnv.get_state_image(obs_state, agent_index, team_indices, enemy_indices, height, width)
        game_vec = PacmanEnv.get_state_vector(obs_state, agent_index, team_indices, enemy_indices, height, width)
        return {"img": game_img, "vec": game_vec}
        
    
    def get_reward(self, initial_state, after_agent_move_state, after_enemy_move_state, picked_illegal_action):
        state = after_enemy_move_state
        state_score = state.get_score() if self.is_red else -state.get_score()
        friendlies_carrying = [state.data.agent_states[self.team_indices[0]].num_carrying,
                                 state.data.agent_states[self.team_indices[1]].num_carrying]
        
        food_list = state.get_red_food().as_list() if self.is_red else state.get_blue_food().as_list()
        my_pos = state.get_agent_state(self.agent_index).get_position()
        if len(food_list) > 2:
            min_dist_to_food = min([self.agents[self.agent_index].distancer.getDistance(my_pos, food) for food in food_list])
            if self.agent_index in self.enemy_indices:
                min_dist_to_food = -min_dist_to_food
        else:
            if self.agent_index in self.enemy_indices:
                min_dist_to_food = 100
            else:
                min_dist_to_food = -100
        
        return state_score + friendlies_carrying[0] + friendlies_carrying[1] - 0.1 * min_dist_to_food
        
    def apply_action(self, state, agent_index, action):
        AgentRules.apply_action(state, action, agent_index)
        AgentRules.check_death(state, agent_index)
        AgentRules.decrement_timer(state.data.agent_states[agent_index])

        # Bookkeeping
        state.data._agent_moved = agent_index
        state.data.score += state.data.score_change
        state.data.timeleft = state.data.timeleft - 1
        return state
    
    def step_game(self, action):
        self.prev_state = self.game.state
        # self.game.state = self.game.state.generate_successor(self.agent_index, action)
        self.game.state = self.apply_action(self.game.state, self.agent_index, action)
        self.game.move_history.append((self.agent_index, action))
        
        self.rules.process(self.game.state, self.game)
        self.agent_index = (self.agent_index + 1) % NUM_AGENTS
    
    # Steps using agent.get_action
    def step_with_cur_agents_action(self):
        cur_agent = self.agents[self.agent_index]
        obs_state = cur_agent.observation_function(self.game.state.deep_copy()) # This has the enemy positions removed
        enemy_action = cur_agent.get_action(obs_state)
        self.step_game(enemy_action)

    def step(self, action):
        initial_state = self.game.state
        initial_index = self.agent_index
        translated_action = ACTION_NAMES[action]
        # TODO: for now, we train the agent to play as both players in the team.
        # Might need to test playing as only one player.
        # This also influences how it will be deployed (as playing for both or just one)
        
        # Our agent's step
        # TODO: find a better way to deal with illegal actions. Otherwise, the agent should learn quickly not to make illegal moves.
        # If this works well, for deployment we will just select a random action if it (still extremely rarely hopefully) picks an illegal action.
        picked_illegal_action = False
        if translated_action not in self.game.state.get_legal_actions(self.agent_index):
            picked_illegal_action = True
            translated_action = np.random.choice(self.game.state.get_legal_actions(self.agent_index))
        self.step_game(translated_action)
        after_agent_move_state = self.game.state
        terminated = self.game.game_over
        
        
        if not terminated:
            # The enemy's step
            self.step_with_cur_agents_action()
            terminated = self.game.game_over
        after_agent_move_state = self.game.state
        
        obs_agent_index = initial_index if terminated else self.agent_index
        obs_state = self.agents[obs_agent_index].observation_function(self.game.state.deep_copy()) # This has the enemy positions removed'
        game_dict = self.get_state_dict(self.agents[obs_agent_index], obs_state, obs_agent_index, self.team_indices, self.enemy_indices, self.height, self.width)
        # TODO: create a BETTER reward function and return reward.
        reward = self.get_reward(initial_state, after_agent_move_state, after_agent_move_state, picked_illegal_action)
        # TODO: find out what truncated and info are and if we need them
        truncated = False
        info = {'legal_actions': self.game.state.get_legal_actions(self.agent_index)} # This might be useful in the future, rn it's useless
        
        return game_dict, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        self.agents = sum([list(el) for el in zip(self.red_agents, self.blue_agents)], [])
        self.game = self.rules.new_game(layout = self.layout, agents = self.agents, display = textDisplay.NullGraphics(), length = self.max_episode_length, mute_agents = False, catch_exceptions = False)
        self.agent_index = self.game.starting_index
        for agent in self.agents:
            agent.register_initial_state(self.game.state.deep_copy())
        # TODO: same as before with the observation
        self.height = self.layout.height
        self.width = self.layout.width

        if self.blue_agents == self.rl_agents:
            print("We are team BLUE")
            self.team_indices = self.game.state.blue_team
            self.enemy_indices = self.game.state.red_team
            self.is_red = False
        else:
            print("We are team RED")
            self.team_indices = self.game.state.red_team
            self.enemy_indices = self.game.state.blue_team
            self.is_red = True

        # If the initial agent is an enemy, we need to step with its action first
        if self.agents[self.agent_index] in self.enemy_agents:
            self.step_with_cur_agents_action()
        
        obs_state = self.agents[self.agent_index].observation_function(self.game.state.deep_copy()) # This has the enemy positions removed
        game_obs = self.get_state_dict(self.agents[self.agent_index], obs_state, self.agent_index, self.team_indices, self.enemy_indices, self.height, self.width)
        
        # TODO: same as before with the info
        info = {'legal_actions': self.game.state.get_legal_actions(self.agent_index)}
        return game_obs, info

    # Probably not needed since we can display the game using the replay method.
    def render(self):
        pass

    # TODO: find out what this is for
    def close(self):
        pass