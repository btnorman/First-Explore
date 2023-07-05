# First-Explore Code


Plots:
- Plots contains the code for reproducing the plots, as well as saved models
- This done via the notebooks. Running all cells in the notebook produces the figures in the paper.

Code:
- darkroom contains the code for the dark treasure room environment
- lte_code contains the code for First-Explore, as well as the Bandit environment

Runs: <br>
The four run folders contain code to replicate the experiments training the first-explore models for the two environments, as well as the always-exploit controls

Each folder contains: <br>
- the .sh script that is used in a slurm environment to launch the python training script on a server
- the .py script that performs the training runs, when passed the appropriate arguments (see the .sh script)
- folders with all the trained models, (saved as run_data.pkl)

The python environment used, (e.g., 'hf' in the .sh scripts), is specified by the requirements.txt file. This environment should be set as the python kernel of the notebooks. Note, this uses jax with GPU support, which can sometimes be tricky to install, e.g., locally on a mac.