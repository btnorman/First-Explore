# First-Explore

This repo reproduces the results from the paper, [First-Explore, then Exploit: Meta-Learning Intelligent Exploration](https://arxiv.org/abs/2307.02276). First-Explore is a general framework for meta-RL in which two context-conditioned policies are trained, one to explore (gather an informative environment rollout based on the current context), and one to exploit (map the current context to high reward behaviour). Each time the policies are used in an environment, the context provided to the policies is all the previous explore rollouts in that environment. By learning two policies, First-Explore decouples Exploration from Exploitation, avoiding the conflict of having to do both simultaneously. This decoupling allows First-Explore to intentionally perform exploration that requires *sacrificing* episode reward (e.g., spending a whole episode training a new skill the agent is bad at, for example practicing with an unfamilair difficult-to-use-but-effective-once-mastered weapon in a fighting game).

As First-Explore is a meta-RL framework, it is trained on a distribution of environments. Training on a distribution allows the policies to learn (via weight updates) how to best do the following: in-context adapt to perform the policy task (exploration or exploitation) based on the prior that an encountered environment is sampled from the training environment distribution. Once trained, the policies then learn about new environments via in-context adaption (with that adaptation to a new environment being the analogue of standard-RL training on a new environment).

Note: this repo is just an example instance of First-Explore. First-Explore is a framework and is applicable to general meta-RL. 

## Repo Structure:
Plots:
- Plots contains the code for reproducing the plots, as well as saved models.
- This done via the notebooks. Running all cells in the notebook produces the figures in the paper.

Code:
- darkroom contains the code for the dark treasure room environment.
- lte_code contains the code for First-Explore, as well as the Bandit environment.

Runs: <br>
The four run folders contain code to replicate the experiments training the first-explore models for the two environments, as well as the always-exploit controls.

Each folder contains: <br>
- the .sh script that is used in a slurm environment to launch the python training script on a server.
- the .py script that performs the training runs, when passed the appropriate arguments (see the .sh script).
- folders with all the trained models, (saved as run_data.pkl).

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
