#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=0-100
#SBATCH --time=00:15:00
#SBATCH --partition=amilan
#SBATCH --output=SlurmFiles/orbit-search-%j-%a.out

module purge

echo "== Load Anaconda =="

module load anaconda

echo "== Activate Env =="

conda activate research

echo "== Run data generation =="

echo "Running orbit search"

 #$SLURM_ARRAY_TASK_ID
srun python /projects/joma5012/FrozenOrbits/FrozenOrbits/Scripts_Orbits/Experiments/MC_search.py cart $SLURM_ARRAY_TASK_ID
# srun python /projects/joma5012/FrozenOrbits/FrozenOrbits/Scripts_Orbits/Experiments/MC_search.py trad $SLURM_ARRAY_TASK_ID
# srun python /projects/joma5012/FrozenOrbits/FrozenOrbits/Scripts_Orbits/Experiments/MC_search.py equi $SLURM_ARRAY_TASK_ID
# srun python /projects/joma5012/FrozenOrbits/FrozenOrbits/Scripts_Orbits/Experiments/MC_search.py mil $SLURM_ARRAY_TASK_ID

wait # Necessary to wait for all processes to finish
echo "== End of Job =="
exit 0