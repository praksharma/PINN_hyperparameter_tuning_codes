#!/bin/bash
#SBATCH --nodes 1
#SBATCH --job-name PINN_training
#SBATCH -o batch_output.log
#SBATCH -e batch_error.log
#SBATCH --gres=gpu:1
#SBATCH --account=scw1901
#SBATCH --partition=accel_ai

#Debug
#echo $RANK
#echo $SLURM_NPROCS
#echo $CUDA_VISIBLE_DEVICES

module load git/2.19.2 # mlflow needs git to play tennis
#env # debug
# python env
source /scratch/s.1915438/modulus_pysdf/modulus_pysdf/bin/activate

srun python run.py -m optimizer.lr=1e-5,2.5e-5,5e-5,7.5e-5,1e-4,2.5e-4,5e-4,7.5e-4,1e-3,2.5e-3,5e-3,7.5e-3,1e-2 custom.arch="DGMArch","FullyConnectedArch","FourierNetArch","SirenArch","ModifiedFourierNetArch","DGMArch"  training.max_steps=10000 +activation='gelu','relu','selu','sin','tanh' +layers=1,5,10,20
