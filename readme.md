# First-Explore

This is the code repository for [First-Explore, then Exploit: Meta-Learning to Solve Hard Exploration-Exploitation Trade-Offs](https://arxiv.org/abs/2307.02276).

Abstract:
> Standard reinforcement learning (RL) agents never intelligently explore like a human (i.e. taking into account complex domain priors and adapting quickly based on previous exploration). Across episodes, RL agents struggle to perform even simple exploration strategies, for example systematic search that avoids exploring the same location multiple times. This poor exploration limits performance on challenging domains. Meta-RL is a potential solution, as unlike standard RL, meta-RL can *learn* to explore, and potentially learn highly complex strategies far beyond those of standard RL, strategies such as experimenting in early episodes to learn new skills, or conducting experiments to learn about the current environment. Traditional meta-RL focuses on the problem of learning to opimally balance exploration and exploitation to maximize the *cumulative reward* of the episode sequence (e.g., aiming to maximize the total wins in a tournament -- while also improving as a player). We identify a new challenge with state-of-the-art cumulative-reward meta-RL methods. When optimal behavior requires exploration that sacrifices immediate reward to enable higher subsequent reward, existing state-of-the-art cumulative-reward meta-RL methods become stuck on the local optimum of failing to explore. Our method, First-Explore, overcomes this limitation by learning two policies: one to solely explore, and one to solely exploit. When exploring requires forgoing early-episode reward, First-Explore significantly outperforms existing cumulative meta-RL methods. By identifying and solving the previously unrecognized problem of forgoing reward in early episodes, First-Explore represents a significant step towards developing meta-RL algorithms capable of human-like exploration on a broader range of domains.

## Setup:
The python environment used, (e.g., 'hf' in the .sh scripts), is specified by the requirements.txt file. This environment should be set as the python kernel of the notebooks. Note, this uses jax with GPU support, which can sometimes be tricky to install, e.g., locally on a mac.

Example Installation for Linux:
```
python3 -m venv [env_name]
source [env_name]/bin/activate
pip install --upgrade pip
pip install jaxlib==0.3.25+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html --only-binary=jaxlib
pip install -r requirements.txt
```

## Repo Structure:

Code:
- darkroom contains the code for the dark treasure room environment
- lte_code contains the code for First-Explore, as well as the Bandits with One Fix Arm environment
- tiny_world contains the code for the Ray Maze environment

Training Scripts, Saved Models and Example of Running an Model:
- tiny_world contains the files for the Ray Maze agents
- bandit_runs contains the files to train the Bandits with One Fix Arm agents
- treasure-room_runs contains the files to train the Dark Treasure Room agents

Control Files:
- control_files contains the code used to train the RL2, VariBAD and HyperX agents
