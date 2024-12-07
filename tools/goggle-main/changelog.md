# Notes

1. Needed to install first torch - requires special torch file before installing requirements
2. Needed to downgrade pip by: pip install 'pip<24.1'
3. The GPU version does not work, need to use CPU as device. Error: 
/opt/dgl/src/array/array.cc:862: Operator DisjointUnionCoo does not support cuda device.
4. Data needs to be normalized and learning rate very small


## Steps for installation that work
1. pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0+cu116 torch-geometric==2.2.0 torch-sparse==0.6.16+pt112cu116 torch-scatter==2.1.0+pt112cu116 -f https://download.pytorch.org/whl/torch_stable.html -f https://data.pyg.org/whl/torch-1.12.0+cu116.html
2. pip install 'pip<24.1'
3. pip install -r requirements.txt
4. pip install --upgrade synthcity