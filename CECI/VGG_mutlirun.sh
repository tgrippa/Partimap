#!/bin/bash
# Submission script for Dragon2
#SBATCH --time=00:30:00 # hh:mm:ss
#
#SBATCH --ntasks=1
#SBATCH --gres="gpu:1"
#SBATCH --mem-per-cpu=2048 # 2GB
#SBATCH --partition=gpu
#
#SBATCH --mail-user=tgrippa@ulb.ac.be
#SBATCH --mail-type=ALL
#
#SBATCH --comment=PARTIMAP
#
#SBATCH --output=CECI/job_outputs/result-%j.txt
#SBATCH --error=CECI/job_outputs/error-%j.txt

module load releases/2020b
module load OpenMPI/4.1.1-GCC-10.3.0
module load TensorFlow
module load scikit-learn
module load h5py
module load matplotlib

srun pip install --user natsort
srun pip install --user keras-tuner --upgrade

srun python SRC/VGG_custom_train.py
