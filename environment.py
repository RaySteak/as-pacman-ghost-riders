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
# import myTeamTrain

IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
FEATURES_LENGTH = 2
NUM_AGENTS = 4

ACTION_NAMES = ['North', 'South', 'West', 'East', 'Stop']
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
        moving_agent_pos = obs_state.data.agent_states[agent_index].get_position()
        teammate_pos = obs_state.data.agent_states[teammate_index].get_position()
        
        # IF they are on top of eachother
        if teammate_pos == moving_agent_pos:
            game_obs[int(teammate_pos[1]), int(teammate_pos[0])] = ord('T') + ord('G')
        else:
            game_obs[int(teammate_pos[1]), int(teammate_pos[0])] = ord('T')
        
        game_obs = T.Resize((IMAGE_HEIGHT, IMAGE_WIDTH), T.InterpolationMode.BILINEAR)(T.ToTensor()(game_obs))
        return np.transpose(game_obs, [1, 2, 0])
    
    @staticmethod
    def get_state_vector(obs_state, agent_index, team_indices, enemy_indices, height, witdth):
        # TODO: use knwoledge of enemy positions (both real and from previous estimates) to filter out the noise (maybe use a Kalman filter or something)
        enemy_dist = np.array([obs_state.agent_distances[enemy_indices[0]], obs_state.agent_distances[enemy_indices[1]]])
        enemy_dist = enemy_dist / (height + witdth) # Normalize distances
        
        return enemy_dist
    
    @staticmethod
    def get_state_dict(obs_state, agent_index, team_indices, enemy_indices, height, width):
        game_img = PacmanEnv.get_state_image(obs_state, agent_index, team_indices, enemy_indices, height, width)
        game_vec = PacmanEnv.get_state_vector(obs_state, agent_index, team_indices, enemy_indices, height, width)
        return {"img": game_img, "vec": game_vec}
        
    
    def get_reward(self, initial_state, after_agent_move_state, after_enemy_move_state, picked_illegal_action):
        # TODO: Confirm whether this assumption is correct: 
        # From what I understand from the codebase, when the BLUE team scores,
        # the total score of the game gets decreased, and when the RED team scores,
        # the total score of the game gets increased. Thus, negative score means
        # a win for BLUE and a positive score means a win for red.
        # TODO: With this implementation, most rewards will be 0, and the agent
        # might care more about achieving immediate rewards ratherr than winning the game.
        # TODO: Find the best course of action for illegal moves.
        # If we return a negative reward, the agent will just stop as it's always the best move.
        # if picked_illegal_action:
            # return -10
        score_dif_agent = after_agent_move_state.get_score() - initial_state.get_score()
        score_dif_enemy = after_enemy_move_state.get_score() - after_agent_move_state.get_score()
        
        weighted_score_dif = score_dif_agent + score_dif_enemy * 0.5
        
        reward = weighted_score_dif
        if self.rl_agents == self.blue_agents:
            reward = -reward
        return reward
        
    
    def step_game(self, action):
        self.prev_state = self.game.state
        self.game.state = self.game.state.generate_successor(self.agent_index, action)
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
        
        obs_state = self.agents[initial_index if terminated else self.agent_index].observation_function(self.game.state.deep_copy()) # This has the enemy positions removed
        game_dict = self.get_state_dict(obs_state, self.agent_index, self.team_indices, self.enemy_indices, self.height, self.width)
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
        else:
            print("We are team RED")
            self.team_indices = self.game.state.red_team
            self.enemy_indices = self.game.state.blue_team

        # If the initial agent is an enemy, we need to step with its action first
        if self.agents[self.agent_index] in self.enemy_agents:
            self.step_with_cur_agents_action()
        
        obs_state = self.agents[self.agent_index].observation_function(self.game.state.deep_copy()) # This has the enemy positions removed
        game_obs = self.get_state_dict(obs_state, self.agent_index, self.team_indices, self.enemy_indices, self.height, self.width)
        
        # TODO: same as before with the info
        info = {'legal_actions': self.game.state.get_legal_actions(self.agent_index)}
        return game_obs, info

    # Probably not needed since we can display the game using the replay method.
    def render(self):
        pass

    # TODO: find out what this is for
    def close(self):
        pass