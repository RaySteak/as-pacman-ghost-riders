# Team Ghost Riders: Pacman Contest

This repository contains our implementation of the Pacman agents in the Pacman Contest competition.

We tried different approaches for the agents, including using the reinforcement learning algorithm Deep QN to learn a policy. However, it didn't result in useful policies.

The second approach we tried is an MCTS (Monte Carlo Tree Search) agent, which is the approach that gave us the best result and that we decided to submit for the competition.

Both agents are contained in the **myTeam.py** file, which is the file which should be use to run the competition with our team.

# Contents

- environment.py: Implementation of the BaseLine 3 reinforcement learning environment we used to train the RL policy.
- myTeamRL.py: The implementation of the RL agent isolated.
- train_RL.ipynb: This is the script we used to train the RL agents. It serves as an experimental environment to tune the agent and visualize the results. 
- logs: Folder containing the logs of the statistics produced when training the RL agent. It can be viewed using TensorBoard.
- **myTeam.py**: This is the file which contains our implementation of the Pacman agents to be submitted to the competition. It contains offensive and defensive MCTS agent, which are the ones instantiated when calling the myTeam.py file when passing it as an argument to the capture.py script.