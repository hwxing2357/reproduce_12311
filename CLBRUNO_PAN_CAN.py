import matplotlib.pyplot as plt
import torch
import umap
from CL_BRUNO import *
import pandas as pd
import pickle

# data preprocessing
torch.manual_seed(31415)
np.random.seed(314159)
# data downloaded from https://xenabrowser.net/datapages/?cohort=TCGA%20Pan-Cancer%20(PANCAN)
PAN_CAN=...
PAN_CAN_meta=...
# PAN_CAN = pd.read_csv('EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena', sep='\t')
# PAN_CAN_meta = pd.read_csv('Survival_SupplementalTable_S1_20171025_xena_sp', sep='\t')

top_N = 2000
PAN_CAN_samples = PAN_CAN.columns[1:].to_numpy()
PAN_CAN_identifies = PAN_CAN['sample'].to_numpy()
PAN_CAN_np = PAN_CAN.drop('sample', axis=1).T.to_numpy()
PAN_CAN = PAN_CAN.drop('sample', axis=1).T
var_order_1 = np.argsort(np.nanvar(PAN_CAN, axis=0))[::-1]
var_order_2 = np.argsort(PAN_CAN.var(axis=0).to_numpy())[::-1]
PAN_CAN_s = PAN_CAN_np[:, var_order_1[:top_N]]
PAN_CAN_identifies_s = PAN_CAN_identifies[var_order_1[:top_N]]

sample_id = [i for i in range(len(PAN_CAN_samples)) if not any(np.isnan(PAN_CAN_s[i])) and
             PAN_CAN_samples[i] in PAN_CAN_meta['sample'].to_numpy()]
PAN_CAN_s = PAN_CAN_s[sample_id]
PAN_CAN_s[PAN_CAN_s == 0] += 1e-2*np.random.randn(np.sum(PAN_CAN_s == 0))

PAN_CAN_samples_s = PAN_CAN_samples[sample_id]
PAN_CAN_label_s = np.array([PAN_CAN_meta[PAN_CAN_meta['sample'] == _]['cancer type abbreviation'].item()
                           for _ in PAN_CAN_samples_s])

PAN_CAN_label_s_lookup = {_: i for i, _ in enumerate(np.unique(PAN_CAN_label_s))}

PAN_CAN_label_s = np.array([PAN_CAN_label_s_lookup[_] for _ in PAN_CAN_label_s])

# splitting the tasks
label_grouping = [[0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11], [12, 13, 14, 15, 16],
                  [17, 18, 19, 20, 21], [22, 23, 24, 25, 26], [27, 28, 29, 30, 31]]

cat_num = np.cumsum([len(_) for _ in label_grouping])

PAN_CAN_data_train, PAN_CAN_data_test = {}, {}
PAN_CAN_label_train, PAN_CAN_label_test = {}, {}

for i, idx in enumerate(label_grouping):
    my_index = np.array([_ in idx for _ in PAN_CAN_label_s])
    n = sum(my_index)
    partition = np.random.permutation(range(n))
    PAN_CAN_data_test[i] = torch.tensor(PAN_CAN_s[my_index][partition[:n//5]], dtype=torch.float)
    PAN_CAN_label_test[i] = torch.tensor(PAN_CAN_label_s[my_index][partition[:n//5]], dtype=torch.long)
    PAN_CAN_data_train[i] = torch.tensor(PAN_CAN_s[my_index][partition[n//5:]], dtype=torch.float)
    PAN_CAN_label_train[i] = torch.tensor(PAN_CAN_label_s[my_index][partition[(n // 5):]], dtype=torch.long)


def find_res(q, test):
    error_rate = {}
    label_pred = {}

    y_dim = cat_num[q]
    for t in range(q+1):
        print(t)
        y_pred = torch.zeros((PAN_CAN_data_test[t].shape[0], y_dim))
        for i in range(PAN_CAN_data_test[t].shape[0]):
            with torch.no_grad():
                temp, _ = test.prediction(PAN_CAN_data_test[t][i], task_id=0)
            y_pred[i] = temp
        my_pred_test = y_pred.argmax(1)
        print('check train')
        y_pred_train = torch.zeros((PAN_CAN_data_train[t].shape[0], y_dim))
        for i in range(PAN_CAN_data_train[t].shape[0]):
            with torch.no_grad():
                temp, _ = test.prediction(PAN_CAN_data_train[t][i], task_id=0)
            y_pred_train[i] = temp
        my_pred_train = y_pred_train.argmax(1)

        error_rate['task{}'.format(t)] = torch.mean(1.0*(my_pred_test != PAN_CAN_label_test[t])).item()
        error_rate['task{}_train'.format(t)] = torch.mean(1.0 * (my_pred_train != PAN_CAN_label_train[t])).item()
        label_pred['task{}'.format(t)] = y_pred
    return {'curr_error_rate': error_rate, 'curr_label_pred': label_pred}


# model initialisation
test_model = CLBruno(x_dim=PAN_CAN_data_test[0].shape[1], y_dim=256, task_dim=256, cond_dim=512,
                     task_num=1, y_cat_num=[len(label_grouping[0])], n_dense_block=6, n_hidden_dense=256,
                     activation=nn.Tanh(), mu_init=0., var_init=1., corr_init=0.1)

# batch 1, initialisation
epoch=200
task_prediction_record = {}
train_loss_0, test_loss_0 = test_model.train_init(PAN_CAN_data_train[0], PAN_CAN_label_train[0],
                                              torch.zeros(len(PAN_CAN_label_train[0]), dtype=torch.long),
                                              batch_size=128, epoch=epoch, weight_decay=1e-3, lr=1e-4)
task_prediction_record[0] = find_res(0, test_model)

# batch 2
train_loss_1, test_loss_1, reg_loss_1 = test_model.train_continual_task(PAN_CAN_data_train[1], PAN_CAN_label_train[1],
                                                                  0, batch_size=128, epoch=200,
                                                                  weight_decay=1e-3, lr=1e-4)
task_prediction_record[1] = find_res(1, test_model)

# batch 3
train_loss_2, test_loss_2, reg_loss_2 = test_model.train_continual_task(PAN_CAN_data_train[2], PAN_CAN_label_train[2],
                                                                  0, batch_size=128, epoch=200,
                                                                  weight_decay=1e-3, lr=7e-5)
task_prediction_record[2] = find_res(2, test_model)

# batch 4
train_loss_3, test_loss_3, reg_loss_3 = test_model.train_continual_task(PAN_CAN_data_train[3], PAN_CAN_label_train[3],
                                                                  0, batch_size=128, epoch=200,
                                                                  weight_decay=1e-3, lr=1e-4)
task_prediction_record[3] = find_res(3, test_model)

# batch 5
train_loss_4, test_loss_4, reg_loss_4 = test_model.train_continual_task(PAN_CAN_data_train[4], PAN_CAN_label_train[4],
                                                                  0, batch_size=128, epoch=160,
                                                                  weight_decay=5e-4, lr=1e-4)
task_prediction_record[4] = find_res(4, test_model)

# batch 6
train_loss_5, test_loss_5, reg_loss_5 = test_model.train_continual_task(PAN_CAN_data_train[5], PAN_CAN_label_train[5],
                                                                  0, batch_size=128, epoch=180,
                                                                  weight_decay=3e-4, lr=1e-4)
task_prediction_record[5] = find_res(5, test_model)
#
# filehandler = open('CL_PANCAN2.pkl', "wb")
# pickle.dump(task_prediction_record, filehandler)
# filehandler.close()


# oracle model
task_prediction_record_oracle = {}
for _ in range(6):
    my_train = torch.cat([PAN_CAN_data_train[i] for i in range(_+1)], 0)
    my_label = torch.cat([PAN_CAN_label_train[i] for i in range(_+1)])
    test_model = CLBruno(x_dim=PAN_CAN_data_test[0].shape[1], y_dim=128, task_dim=128, cond_dim=256,
                         task_num=1, y_cat_num=[np.cumsum([len(__) for __ in label_grouping])[_]], n_dense_block=6, n_hidden_dense=256,
                         activation=nn.Tanh(), mu_init=0., var_init=1., corr_init=0.1)
    train_loss_0, test_loss_0 = test_model.train_init(my_train, my_label,
                                                  torch.zeros(my_train.shape[0], dtype=torch.long),
                                                  batch_size=128, epoch=200, weight_decay=1e-3, lr=1e-4)
    task_prediction_record_oracle[_] = find_res(_, test_model)

# filehandler = open('CL_PANCAN_oracle2.pkl', "wb")
# pickle.dump(task_prediction_record_oracle, filehandler)
# filehandler.close()


# visualising knowledge retention
file = open("CL_PANCAN.pkl",'rb')
task_prediction_record = pickle.load(file)
file.close()
file = open("CL_PANCAN_oracle.pkl",'rb')
task_prediction_record_oracle = pickle.load(file)
file.close()

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
legend_elements = [Line2D([0], [0], marker='o', color='k', lw=2, label='CIL'),
                   Line2D([0], [0], marker='^', color='k', lw=2, label='Oracle', linestyle='dotted'),
                   Patch(facecolor='orchid', edgecolor='orchid', label='Group 0'),
                   Patch(facecolor='grey', edgecolor='grey', label='Group 1'),
                   Patch(facecolor='darkcyan', edgecolor='darkcyan', label='Group 2'),
                   Patch(facecolor='turquoise', edgecolor='turquoise', label='Group 3'),
                   Patch(facecolor='dodgerblue', edgecolor='dodgerblue', label='Group 4'),
                   Patch(facecolor='darkviolet', edgecolor='darkviolet', label='Group 5'),]

colors = ['orchid', 'grey', 'darkcyan', 'turquoise', 'dodgerblue', 'darkviolet']
fig, ax = plt.subplots()
for i in range(6):
    loss = np.array([task_prediction_record[_]['curr_error_rate']['task{}'.format(i)] for _ in range(i, 6)])
    loss_oracle = np.array([task_prediction_record_oracle[_]['curr_error_rate']['task{}'.format(i)] for _ in range(i, 6)])
    ax.plot(range(i, 6), loss, marker='o', markersize=5, c=colors[i], alpha=0.7)
    ax.plot(range(i, 6), loss_oracle, marker='^', markersize=5, c=colors[i], alpha=0.6, linestyle='dotted')
ax.set_xlabel('Incremental step')
ax.set_ylabel('Misclassification rate')
ax.set_title('PANCAN data under CIL scenario')
ax.legend(handles=legend_elements, loc='upper left')
fig.set_size_inches(5,4.5)
plt.savefig('PANCAN_acc.png')


