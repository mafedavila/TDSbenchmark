# Create environment

conda create -n tabsyn python=3.10
conda activate tabsyn

conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia

## Install other dependencies

pip install -r requirements.txt

## Install dependencies for GOGGLE

pip install  dgl -f https://data.dgl.ai/wheels/cu117/repo.html

pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.1+cu117.html


### Create another environment for the quality metric (package "synthcity")

conda create -n synthcity python=3.10
conda activate synthcity

pip install synthcity
pip install category_encoders

# Data preprocessing

Just like original repo

# Training

For Tabsyn, use the following command for training:

## train VAE first
python main.py --dataname adult --method vae --mode train

## after the VAE is trained, train the diffusion model
python main.py --dataname adult --method tabsyn --mode train

## Synthesis

python main.py --dataname adult --method tabsyn --mode sample
The default save path is "synthetic/[NAME_OF_DATASET]/[METHOD_NAME].csv"


# Notes
Does not work with datasets which have no categorical columns.
Preprocessing is extensive, including a dataset.json with metadata.
The model is not implemented as usual in ML, where import is done. Difficult to manipulate within jupyter notebook. Need to run command and it saves in another folders, etc...