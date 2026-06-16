import os
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from model import PW_classifier
import pdb
np.set_printoptions(suppress=True)

def norm(dt):
    Min = np.min(dt)
    Max = np.max(dt)
    dt = (dt - Min) / (Max - Min)
    return dt


model = PW_classifier(in_dims=70, gcn_layer_num=1, gcn_config=None)
model.load_weights('./output/model/0.9358039760847205.weights')


data_directory = '/opt/data/share/120106022644/protein/datasets/pipgcn/DB5/test.pkl'
_, test_data = pickle.load(open(data_directory, 'rb'))

idx = 0
for data in test_data:
    if data['complex_code'] == "3V6Z":
        values = np.zeros((len(data['l_vertex']), len(data['r_vertex'])))
        pos_idxs = data["label"][data["label"][:, 2] == 1, :2]
        values[pos_idxs.T.tolist()] = 1

        preds = model(data['l_vertex'], data['l_hood_indices'].squeeze(), data['l_edge'], 
                        data['r_vertex'], data['r_hood_indices'].squeeze(), data['r_edge'], data["label"], False)
        
        # [sample[idx][0]:sample[idx][1], sample[idx][2]:sample[idx][3]]
        
        # values = np.matmul(np.max(np.mean(preds[1][0].numpy(), axis=0), axis=1)[:,None], np.transpose(np.max(np.mean(preds[1][1].numpy(), axis=0), axis=1)[:,None]))
        
        
        # bins = [0.326, 0.358, 0.390, 0.422, 0.454, 0.486]
        # indices=np.digitize(dt ,bins)
        # atten_l = np.mean(norm(np.mean(preds[1][1].numpy(), axis=0)), axis=0)
        # atten_r = np.mean(norm(np.mean(preds[1][0].numpy(), axis=0)), axis=0)
        pdb.set_trace()

        # dt = np.matmul(atten_l[:, None], np.transpose(atten_r[:, None]))
        dt = norm(np.mean(preds[1][0].numpy(), axis=0)) +  np.transpose(norm(np.mean(preds[1][1].numpy(), axis=0)))
        dt = norm(dt)
        dt[dt < 0.7] = 0
        # dt[dt > 0.95] = 0

        
        ax = sns.heatmap(dt, cmap="viridis", xticklabels=False, yticklabels=False)
        figure = ax.get_figure()
        figure.savefig("./pic/{:}_a.png".format(data['complex_code']))
        pdb.set_trace()
        plt.close()
        idx += 1
    # pdb.set_trace()
    