
# arg passing
import argparse

from inspect import Parameter
from itertools import combinations
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

class C(list):
    def __init__(self, arg, name=None):
        self.name = name
        for obj in arg:
            if isinstance(obj, C):
                raise Exception("merge these nested choices")
        super().__init__(arg)

# accepts lists and dictionaries
def set_run_parameters(d, n, chosen=None):
    if chosen is None:
        chosen = dict()
    if isinstance(d, dict):
        keys = sorted(d.keys())
    elif isinstance(d, list):
        keys = range(len(d))
    else:
        raise Exception
    for key in keys:
        if isinstance(d[key], C):
            dk = d[key]
            if dk.name in chosen:
                print('get', dk.name, chosen[dk.name])
                d[key] = chosen[dk.name]
            else:
                k = n % len(d[key])
                n = n // len(d[key])
                d[key] = d[key][k]
                if dk.name is not None:
                    print('set', dk.name, d[key])
                    chosen[dk.name] = d[key]
        if isinstance(d[key], (list, dict)):
            n = set_run_parameters(d[key], n, chosen=chosen)
    return n

def count(d, chosen=None):
    if chosen is None:
        chosen = dict()
    n = 1
    if isinstance(d, dict):
        keys = sorted(d.keys())
    elif isinstance(d, list):
        keys = range(len(d))
    else:
        return 1
    for key in keys:
        dk = d[key]
        if isinstance(d[key], C) and dk.name not in chosen:
            if dk.name is not None and dk.name not in chosen:
                chosen[dk.name] = True
            n *= len(d[key])
            min_ = min(map(count, d[key]))
            max_ = max(map(count, d[key]))
            if min_ != max_:
                print("wonky shape!")
            n *= max(map(lambda arg: count(arg, chosen=chosen), d[key]))
        elif isinstance(d[key], (list, dict)):
            n *= count(d[key], chosen=chosen)
    return n

def get_args(num_options):
    # put experiment description here
    parser = argparse.ArgumentParser(description="Experiment to compare clipped and unclipped reward")
    parser.add_argument('--SBATCHID', type=int, help=(
f'''the number of the parameter combination to run.
there are {num_options} options, from 0 to {num_options-1}'''))
    parser.add_argument('--run_ID', type=str, help=(
'''specify a unique identifier for the runs, for wandb logging
this will be combined with SBATCHID to give an ID for the current run'''))
    parser.add_argument('--group', type=str, help="a group name for the experiment")
    parser.add_argument('--SLURM_JOB_ID', type=str, help="the SLURM_JOB_ID, used for checkpointing")
    args = parser.parse_args()
    return args

