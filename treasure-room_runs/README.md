# Running on the Treasure Room

The treasure-room_first-explore.py is edited to set the range of run parameters, which are set at the top of the script. The file structure of the run data (logs and saved models) then record the parameters specified in the .py script.

The python file can be run using

    python treasure-room_first-explore.py --SBATCHID $SLURM_ARRAY_TASK_ID --run_ID [name prefix of each run] --group [run group name for wanbd] --SLURM_JOB_ID $SLURM_JOB_ID
    
The above script is formatted for use with SLURM - but should be easy to adapt to other ways of running the script.