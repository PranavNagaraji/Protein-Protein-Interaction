# Intra-Inter Graph Representation Learning for Protein-Protein Binding Sites Prediction 

# Framework Graph

![image](https://github.com/IIGRLzwt/IIGRL/assets/97393672/b5189f7a-f608-4bda-b567-47ac444d1849)

The framework of our proposed Intra-Inter Graph Representation Learning (IIGRL) for protein-protein binding sites prediction. The upper is intra-graph representation learning, which is designed to improve node/residue representation within protein graph. The bottom is inter-graph representation learning, exploring to propagate information cross protein to further enrich node representation.

# Description
For the intra-graph learning, we propose to maximize the mutual information between the local node representation and global graph summary, thus encouraging the global information encoded into node representation in the protein graph. For the inter-graph learning, we explore fusing the two separate ligand and receptor graphs as a whole graph, learn the affinity between residues/nodes of different proteins, and propagate the information to each other, which effectively captures inter-protein information and further enhances the discrimination of the node pairs.

# Requirements
```
idna==3.4
importlib-metadata==4.8.3
joblib==1.1.1
Keras-Preprocessing==1.1.2
memory-profiler==0.61.0
numpy==1.18.5
oauthlib==3.2.2
opt-einsum==3.3.0
protobuf==3.19.6
psutil==5.9.5
pyasn1==0.4.8
pyasn1-modules==0.2.8
python-version==0.0.2
requests==2.27.1
requests-oauthlib==1.3.1
rsa==4.9
tensorboard==2.10.1
tensorboard-data-server==0.6.1
tensorboard-plugin-wit==1.8.1
tensorflow-estimator==2.2.0
tensorflow-gpu==2.2.0
termcolor==1.1.0
threadpoolctl==3.1.0
typing_extensions==4.1.1
urllib3==1.26.15
Werkzeug==2.0.3
wrapt==1.15.0
zipp==3.6.0
```
# Datasets 
We provide DB5 dataset in Releases. We provide the training weight of DB5 under the model folder.
# Training
```
bash run.sh
```
# Test
```
python test.py
```
