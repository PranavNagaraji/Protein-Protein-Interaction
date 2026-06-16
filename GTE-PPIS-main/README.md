# GTE-PPIS  
GTE-PPIS is a novel structure-based protein-protein interaction site predictor that uses two branches, graph transformer and equivariant graph neural network, to capture amino acid information and complete site prediction tasks.

# System requirement  
GTE-PPIS is developed under Linux environment with:  
python  3.10.14  
numpy  1.26.4  
pandas  2.2.2 
scikit-learn  1.5.0   
pytorch  2.4.1   
dgl 2.4.0  

# Dataset and Feature  
The datasets used in this study (Train_335, Test_60, Test_315-28 and UBtest_31-6) are stored in ./Dataset in python dictionary format:  
```
Dataset[ID] = [seq, label]
```
The  normalized feature matrixes PSSM(L * 20), HMM(L * 20), DSSP(L * 14) and resAF(L * 7) are stored in ./Feature in numpy format.  

Due to GitHub limitations, the distance maps(L * L) data file is stored on Google Drive:
[distance maps](https://drive.google.com/drive/folders/1UKX1dIrzrEPQpGyYcvIJlrm6KAKvNEpp?usp=drive_link)

The protein PDB files are stored on Google Drive:
[PDB files](https://drive.google.com/drive/folders/1eTjFtxsP4mnzyg5CXs96c3w4AxzYNW5N?usp=drive_link)
  
# Running GTE-PPIS 
Train the model with default parameters:  
```
python train.py
```  
Test the model you just trained on the test sets:  
```
python test.py
```
You can adjust the parameters via EGNN_model.py and GraphTransformer_Block.py  

