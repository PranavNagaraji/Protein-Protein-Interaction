import os
import pandas as pd
from torch.autograd import Variable
from sklearn import metrics
from torch.utils.data import DataLoader

from data_generator import *
from EGNN_model import *
from final_model import *
from GraphTransformer_Block import *

Dataset_Path = "./Dataset/"
Model_Path = "./Log/2024-12-09-19-12-36/model/"


def evaluate(model, data_loader):
    model.eval()

    epoch_loss = 0.0
    n = 0
    valid_pred = []
    valid_true = []
    pred_dict = {}

    for data in data_loader:
        with torch.no_grad():
            sequence_names, _, labels, node_features, G_batch, adj_matrix, xyz_feats, edges, edge_att, edge_feat = data

            if torch.cuda.is_available():
                node_features = Variable(node_features.cuda().float())
                adj_matrix = Variable(adj_matrix.cuda())
                G_batch.edata['ex'] = Variable(G_batch.edata['ex'].float())
                G_batch = G_batch.to(torch.device('cuda:0'))
                xyz_feats = Variable(xyz_feats.cuda().float())
                edges = Variable(edges.cuda())
                edge_att = Variable(edge_att.cuda().float())
                edge_feat = Variable(edge_feat.cuda().float())
                y_true = Variable(labels.cuda())

            else:
                node_features = Variable(node_features.float())
                adj_matrix = Variable(adj_matrix)
                xyz_feats = Variable(xyz_feats.float())
                edges = Variable(edges)
                edge_att = Variable(edge_att.float())
                edge_feat = Variable(edge_feat.float())
                y_true = Variable(labels)
                G_batch.edata['ex'] = Variable(G_batch.edata['ex'].float())

            adj_matrix = torch.squeeze(adj_matrix)
            y_true = torch.squeeze(y_true)
            y_true = y_true.long()

            y_pred = model(node_features, xyz_feats, edges, edge_att, edge_feat, adj_matrix)

            loss = model.criterion(y_pred, y_true)
            softmax = torch.nn.Softmax(dim=1)
            y_pred = softmax(y_pred)
            y_pred = y_pred.cpu().detach().numpy()
            y_true = y_true.cpu().detach().numpy()
            valid_pred += [pred[1] for pred in y_pred]
            valid_true += list(y_true)
            pred_dict[sequence_names[0]] = [pred[1] for pred in y_pred]

            epoch_loss += loss.item()
            n += 1
    epoch_loss_avg = epoch_loss / n

    return epoch_loss_avg, valid_true, valid_pred, pred_dict


def analysis(y_true, y_pred, best_threshold = None):
    if best_threshold == None:
        best_f1 = 0
        best_threshold = 0
        for threshold in range(0, 100):
            threshold = threshold / 100
            binary_pred = [1 if pred >= threshold else 0 for pred in y_pred]
            binary_true = y_true
            f1 = metrics.f1_score(binary_true, binary_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

    binary_pred = [1 if pred >= best_threshold else 0 for pred in y_pred]
    binary_true = y_true

    # binary evaluate
    binary_acc = metrics.accuracy_score(binary_true, binary_pred)
    precision = metrics.precision_score(binary_true, binary_pred)
    recall = metrics.recall_score(binary_true, binary_pred)
    f1 = metrics.f1_score(binary_true, binary_pred)
    AUC = metrics.roc_auc_score(binary_true, y_pred)
    precisions, recalls, thresholds = metrics.precision_recall_curve(binary_true, y_pred)
    AUPRC = metrics.auc(recalls, precisions)
    mcc = metrics.matthews_corrcoef(binary_true, binary_pred)

    results = {
        'binary_acc': binary_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'AUC': AUC,
        'AUPRC': AUPRC,
        'mcc': mcc,
        'threshold': best_threshold
    }
    return results


def test(test_dataframe, psepos_path):
    all_metrics = {
        'binary_acc': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'AUC': [],
        'AUPRC': [],
        'mcc': [],
        'threshold': []
    }
        
    test_loader = DataLoader(dataset=ProDataset(dataframe=test_dataframe,psepos_path=psepos_path), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=graph_collate)

    for model_name in sorted(os.listdir(Model_Path)):
        print(model_name)
        model = FinalModel(INPUT_DIM, HIDDEN_DIM, FLITER_DIM, OUTPUT_SIZE, DROPOUT, LAYER)
        if torch.cuda.is_available():
            model.cuda()
        model.load_state_dict(torch.load(Model_Path + model_name, map_location='cuda:0', weights_only=True))

        epoch_loss_test_avg, test_true, test_pred, pred_dict = evaluate(model, test_loader)

        result_test = analysis(test_true, test_pred)

        for key in all_metrics:
            all_metrics[key].append(result_test[key])

        print("========== Evaluate Test set ==========")
        print("Test loss: ", epoch_loss_test_avg)
        print("Test binary acc: ", result_test['binary_acc'])
        print("Test precision:", result_test['precision'])
        print("Test recall: ", result_test['recall'])
        print("Test f1: ", result_test['f1'])
        print("Test AUROC: ", result_test['AUC'])
        print("Test MCC: ", result_test['mcc'])
        print("Test AUPRC: ", result_test['AUPRC'])
        print("Threshold: ", result_test['threshold'])
        print()

    return all_metrics

def test_one_dataset(dataset, psepos_path):
    IDs, sequences, labels = [], [], []
    for ID in dataset:
        IDs.append(ID)
        item = dataset[ID]
        sequences.append(item[0])
        labels.append(item[1])
    test_dic = {"ID": IDs, "sequence": sequences, "label": labels}
    test_dataframe = pd.DataFrame(test_dic)
    all_metrics = test(test_dataframe, psepos_path)

    average_metrics = {key: np.mean(values[:5]) for key, values in all_metrics.items()}
    print("========== Cross-Validation Results ==========")
    print("Average binary acc: ", average_metrics['binary_acc'])
    print("Average precision: ", average_metrics['precision'])
    print("Average recall: ", average_metrics['recall'])
    print("Average f1: ", average_metrics['f1'])
    print("Average AUROC: ", average_metrics['AUC'])
    print("Average MCC: ", average_metrics['mcc'])
    print("Average AUPRC: ", average_metrics['AUPRC'])
    print("Average threshold: ", average_metrics['threshold'])
    print()


def main():
    with open(Dataset_Path + "Test_60.pkl", "rb") as f:
        Test_60 = pickle.load(f)

    with open(Dataset_Path + "Test_315-28.pkl", "rb") as f:
        Test_315_28 = pickle.load(f)

    with open(Dataset_Path + "UBtest_31-6.pkl", "rb") as f:
        UBtest_31_6 = pickle.load(f)

    Btest_31_6 = {}
    with open(Dataset_Path + "bound_unbound_mapping31-6.txt", "r") as f:
        lines = f.readlines()[1:]
    for line in lines:
        bound_ID, unbound_ID, _ = line.strip().split()
        Btest_31_6[bound_ID] = Test_60[bound_ID]

    # print(f"Btest_31_6 keys: {list(Btest_31_6.keys())}")
    # raise

    Test60_psepos_Path = '/pubssd/PPIs/Feature/psepos/Test60_psepos_SC.pkl'
    Test315_28_psepos_Path = '/pubssd/PPIs/Feature/psepos/Test315-28_psepos_SC.pkl'
    Btest31_psepos_Path = '/pubssd/PPIs/Feature/psepos/Test60_psepos_SC.pkl'
    UBtest31_28_psepos_Path = '/pubssd/PPIs/Feature/psepos/UBtest31-6_psepos_SC.pkl'

    print("Evaluate GraphPPIS on Test_60")
    test_one_dataset(Test_60, Test60_psepos_Path)

    # print("Evaluate GraphPPIS on Test_315-28")
    # test_one_dataset(Test_315_28, Test315_28_psepos_Path)

    # print("Evaluate GraphPPIS on Btest_31-6")
    # test_one_dataset(Btest_31_6, Btest31_psepos_Path)

    # print("Evaluate GraphPPIS on UBtest_31-6")
    # test_one_dataset(UBtest_31_6, UBtest31_28_psepos_Path)


if __name__ == "__main__":
    main()
