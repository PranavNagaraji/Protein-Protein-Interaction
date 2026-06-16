import os
from re import S
import sys
import time
import pickle
import tensorflow as tf
# tf.enable_eager_execution()
import numpy as np
from sklearn.metrics import roc_curve, auc,  recall_score, precision_score, f1_score, average_precision_score, matthews_corrcoef, precision_recall_curve, roc_auc_score
from configuration import data_directory, output_directory, seeds, printt, GCN_layers
from train_test import TrainTest
from model import PW_classifier, Weight_Cross_Entropy
from results_processor import ResultsProcessor
import tensorflow.keras.backend as K
import pdb
import matplotlib.pyplot as plt
tf.get_logger().setLevel('ERROR')

def plot_eval_predictions(labels, predictions, path="figure"):

    pos_phat = predictions[labels == 1]
    neg_phat = predictions[labels == 0]

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle("Distribution of Predictions")
    ax1.hist(pos_phat)
    ax1.set_xlim(0, 1)
    ax1.set_title("Positive")
    ax1.set_xlabel("p-hat")
    ax2.hist(neg_phat)
    ax2.set_xlim(0, 1)
    ax2.set_title("Negative")
    ax2.set_xlabel("p-hat")
    plt.savefig(str(path) + ".phat_dist.png")
    plt.close()

def roc(data):
    import pdb
    # pdb.set_trace()
    # scores = [prot[0] for prot in data]
    # labels = [prot[1] for prot in data]
    fprs = []
    tprs = []
    roc_aucs = []
    Recall =[]
    Spec =[]
    F1 =[]
    Pre =[]
    MCC =[]
    Aupr = []
    # Aupr1 = []
    # Auproc1=[]
    # pdb.set_trace()
    i = 0
    for prot in data:
        s = prot[0].cpu()
        l = (prot[1]+1)/2
        fpr, tpr, _ = roc_curve(l, s)
        roc_auc = auc(fpr, tpr)
        s = np.where(np.array(s) < 0.5, 0, 1)
        # s = [1 if p >= 0.5 else 0 for p in s]
        recall = recall_score(l, s)
        specificity = recall_score(l, s, pos_label=0)
        # Precision_score = average_precision_score(l, s)
        Precision_score = precision_score(l, s)
        plot_eval_predictions(l,s,i)
        i +=1
        mcc = matthews_corrcoef(l, s)
        f1 = f1_score(l, s)

        bb, aa, _ = precision_recall_curve(l, s)
        aupr = auc(aa, bb)
        Aupr.append(aupr)
        
        # aupr1 = average_precision_score(l,s)
        # auroc1 = roc_auc_score(l,s)
        # Aupr1.append(aupr1)
        # Auproc1.append(auroc1)
        Recall.append(recall)
        Spec.append(specificity)
        F1.append(f1)
        fprs.append(fpr)
        tprs.append(tpr)
        Pre.append(Precision_score)
        MCC.append(mcc)
        roc_aucs.append(roc_auc)
    auc_prot_med = np.median(roc_aucs)
    auc_prot_ave = np.mean(roc_aucs)
    auc_prot_max = np.max(roc_aucs)
    auc_prot_min = np.min(roc_aucs)
    # print (roc_aucs)
    print(" average protein auc: {:0.3f}".format( auc_prot_ave))
    print(" median protein auc: {:0.3f}".format( auc_prot_med))
    print(" max protein auc: {:0.3f}".format( auc_prot_max))
    print(" min protein auc: {:0.3f}".format( auc_prot_min))
    print("median is Recall:", np.median(Recall))
    print("median is spec:", np.median(Spec))
    print("median is F1: ", np.median(F1))
    print("median is MCC: ", np.median(MCC))
    print("median is Precision: ", np.median(Pre))
    # print(Aupr)
    print("median is aupr: ", np.median(Aupr))
    # print("median is aupr: ", np.median(Aupr1))
    # print("median is aupr: ", np.median(Auproc1))
    print("+++++++++++++++++++++++++++++++++++++++++++++")
    print("mean is Recall:", np.mean(Recall))
    print("mean is spec:", np.mean(Spec))
    print("mean is F1: ", np.mean(F1))
    print("mean is MCC: ", np.mean(MCC))
    print("mean is Precision: ", np.mean(Pre))
    return auc_prot_med

os.environ['CUDA_VISIBLE_DEVICES']='1'

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)



printt("Load train data")
train_data_file = "/data3/xugongping/graduation/datasets/DB5/train.pkl"
_, train_data = pickle.load(open(train_data_file, 'rb'))
printt("Load test data")
test_data_file = "/data3/xugongping/graduation/datasets/DB5/test.pkl"
_, test_data = pickle.load(open(test_data_file, 'rb'))

data = {"train": train_data, "test": test_data}


for i, seed_pair in enumerate(seeds):
    for j, gcn_layer in enumerate(GCN_layers):
        K.clear_session()
        printt("rep{:}/layer_{:}".format(i, j + 1))
        # set tensorflow and numpy seed
        tf.random.set_seed(seed_pair['tf_seed'])
        np.random.seed(int(seed_pair['np_seed']))
        
        printt("build model")
        in_dims = 70 
        learning_rate = 0.1
        
        pn_ratio = 0.1 

        model = PW_classifier(in_dims=in_dims, gcn_layer_num=j + 1, gcn_config=gcn_layer[j + 1])
        cerition = Weight_Cross_Entropy(pn_ratio=pn_ratio)
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

        # model.load_weights('/data3/wl/IIGRL/output/model/models/my_checkpoint')
        model.load_weights('/data3/wl/IIGRL/output/model/normal_2/0.8953909076608463.weights')

        prot_perm = np.random.permutation(len(data["test"]))
        print(len(data["test"]))
        # loop through each protein
        sum_loss = 0.
        n_batch = 0
        pred_label = []
        begin_time = time.time()
        for protein in data["test"]:
            prot_data = protein
            pair_examples = prot_data["label"]

            labels = pair_examples[:,2]

            l_lbl_1 = tf.ones(prot_data['l_vertex'].shape[0])
            l_lbl_2 = tf.zeros(prot_data['l_vertex'].shape[0])
            l_lbl = tf.concat((l_lbl_1, l_lbl_2), axis=0)

            r_lbl_1 = tf.ones(prot_data['r_vertex'].shape[0])
            r_lbl_2 = tf.zeros(prot_data['r_vertex'].shape[0])
            r_lbl = tf.concat((r_lbl_1, r_lbl_2), axis=0)

            
            preds, _, _ = model(prot_data['l_vertex'], prot_data['l_hood_indices'].squeeze(), prot_data['l_edge'], 
                            prot_data['r_vertex'], prot_data['r_hood_indices'].squeeze(), prot_data['r_edge'], 
                            pair_examples, False)
            
                
            # loss = cerition(preds, labels, (l_lbl, r_lbl))
            pred_label.append([tf.squeeze(preds), labels])
            # sum_loss += loss
        # printt("epoch avg loss {:}".format(sum_loss / len(data)))
        # print("roc指标：")
        end_time = time.time()
        print("time:"+ str(end_time-begin_time))
        print("time:"+ str((end_time-begin_time)/55.0))
        Roc = roc(pred_label)
    break
        

    
