#!/bin/bash
#SBATCH -p a100_short,a100_long
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100GB
#SBATCH --time=48:00:00
#SBATCH --job-name=jupyter
#SBATCH --exclude=a100-4020
#SBATCH --output jupyter-log/jupyter-notebook-%J.log

# Thank you, Yale Center for Research Computing, for this script
# modified for use on BigPurple in NYU Langone Medical Center by Paul Glick

# get tunneling info
XDG_RUNTIME_DIR=""
port=8890
node=$(hostname -s)
user=$(whoami)
#cluster=$(hostname -f | awk -F"." '{print $2}')
case "$SLURM_SUBMIT_HOST" in
bigpurple-ln1)
 login_node=bigpurple1.nyumc.org
 ;;
bigpurple-ln2)
 login_node=bigpurple2.nyumc.org
 ;;
bigpurple-ln3)
 login_node=bigpurple3.nyumc.org
 ;;
bigpurple-ln4)
 login_node=bigpurple4.nyumc.org
 ;;
esac
# print tunneling instructions jupyter-log
echo -e "
MacOS or linux terminal command to create your ssh tunnel:
ssh -N -L ${port}:${node}:${port} ${user}@${login_node}

For more info and how to connect from windows,
   see research.computing.yale.edu/jupyter-nb
Here is the MobaXterm info:

Forwarded port:same as remote port
Remote server: ${node}
Remote port: ${port}
SSH server: ${login_node}
SSH login: $user
SSH port: 22
Use a Browser on your local machine to go to:
http://localhost:${port}  (prefix w/ https:// if using password)
Use the token string from the URL printed below and add it to the URL above
"

# load modules or conda environments here
# e.g. farnam:s
#module add anaconda2/gpu/5.2.0
#module add anaconda3/cpu/5.3.1

module load gcc/8.1.0
module load rust

# STEP #1: replace the bashrc with your bashrc
source /gpfs/home/wz1492/.bashrc

# Set custom Jupyter runtime directory to avoid permission issues
export JUPYTER_RUNTIME_DIR=/gpfs/data/shenlab/wz1492/jupyter-log

# STEP #2: activate the conda env
conda activate rag

# STEP #3: cd to your workspace under /gpfs/data/shenlab/xxxx
cd /gpfs/data/shenlab/wz1492/my-rag-project

# STEP #4: start the jupyter notebook
python app.py