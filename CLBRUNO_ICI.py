import matplotlib.pyplot as plt
import torch
import umap
from CL_BRUNO import *
import pandas as pd

# data available at https://isb-cgc.shinyapps.io/iatlas/
torch.manual_seed(31415)
ICI_data = pd.read_csv('ICI_top_500_DR.csv') .sample(frac=1., random_state=314159, ignore_index=True)
print(ICI_data)
ICI_data['ICI_Rx'] = ICI_data['ICI_Rx'].fillna('NA')
ICI_unique_task = ['Atezo', 'Nivo', 'Pembro', 'NA', 'Ipi', 'Ipi + Pembro', 'Ipi + Nivo']
ICI_task = np.zeros(ICI_data.shape[0])
ICI_task_look_up = {}
for i, j in enumerate(ICI_unique_task):
    ICI_task[j == ICI_data['ICI_Rx']] = i
    ICI_task_look_up[i] = j

ICI_task = ICI_task.astype(int)

ICI_label_look_up = {}
for i, j in ICI_task_look_up.items():
    sample = ICI_data['Cancer_tissue'][ICI_data['ICI_Rx'] == ICI_task_look_up[i]]
    ICI_label_look_up[i] = {}
    for k, l in enumerate(sample.unique()):
        ICI_label_look_up[i][k] = l

ICI_label = np.array([list(ICI_label_look_up[ICI_task[i]].values()).index(ICI_data['Cancer_tissue'][i])
                     for i in range(ICI_data.shape[0])])

ICI_task_label = ICI_data['ICI_Rx']
ICI_tissue_label = ICI_data['Cancer_tissue']
ICI_data = ICI_data.drop(['ICI_Rx', 'Cancer_tissue'], axis=1).to_numpy()

task_sample_size = [sum(ICI_task == i) for i in range(7)]
task_categories = [len(ICI_label_look_up[_]) for _ in range(7)]

print([ICI_tissue_label[ICI_task == i].unique() for i in range(7)])

grouped_task_test = [torch.tensor(ICI_task[ICI_task == i][:sum(ICI_task == i)//5], dtype=torch.long) for i in ICI_task_look_up.keys()]
grouped_task_train = [torch.tensor(ICI_task[ICI_task == i][sum(ICI_task == i)//5:], dtype=torch.long) for i in ICI_task_look_up.keys()]
grouped_label_test = [torch.tensor(ICI_label[ICI_task == i][:sum(ICI_task == i)//5], dtype=torch.long) for i in ICI_task_look_up.keys()]
grouped_label_train = [torch.tensor(ICI_label[ICI_task == i][sum(ICI_task == i)//5:], dtype=torch.long) for i in ICI_task_look_up.keys()]
grouped_data_test = [torch.tensor(ICI_data[ICI_task == i][:sum(ICI_task == i)//5], dtype=torch.float) for i in ICI_task_look_up.keys()]
grouped_data_train = [torch.tensor(ICI_data[ICI_task == i][sum(ICI_task == i)//5:], dtype=torch.float) for i in ICI_task_look_up.keys()]

def find_res(q, test):
    error_rate = {}
    label_pred = {}
    task_pred = {}
    for t in range(q+1):
        print(t)
        y_pred, t_pred = torch.zeros((grouped_data_test[t].shape[0], task_categories[t])), torch.zeros((grouped_data_test[t].shape[0], q+1))
        for i in range(grouped_data_test[t].shape[0]):
            with torch.no_grad():
                temp, t_pred[i] = test.prediction(grouped_data_test[t][i])
            y_pred[i] = temp[t]
        my_pred_test = y_pred.argmax(1)

        y_pred = torch.zeros((grouped_data_train[t].shape[0], task_categories[t]))
        for i in range(grouped_data_train[t].shape[0]):
            with torch.no_grad():
                temp, _ = test.prediction(grouped_data_train[t][i])
            y_pred[i] = temp[t]
        my_pred_train = y_pred.argmax(1)

        task_pred['task{}'.format(t)] = t_pred
        error_rate['task{}'.format(t)] = torch.mean(1.0*(my_pred_test != grouped_label_test[t])).item()
        error_rate['task{}_train'.format(t)] = torch.mean(1.0 * (my_pred_train != grouped_label_train[t])).item()
        label_pred['task{}'.format(t)] = y_pred
    return {'curr_task_pred': task_pred, 'curr_error_rate': error_rate, 'curr_label_pred': label_pred}

# model initialisation
test_model = CLBruno(x_dim=ICI_data.shape[1], y_dim=128, task_dim=128, cond_dim=256,
                     task_num=1, y_cat_num=[task_categories[0]], n_dense_block=6, n_hidden_dense=256,
                     activation=nn.Tanh(), mu_init=0., var_init=1., corr_init=0.1)
# task 1
epoch=100
task_prediction_record = {}
train_loss, test_loss = test_model.train_init(grouped_data_train[0],
                                              grouped_label_train[0], grouped_task_train[0], batch_size=128,
                                              epoch=epoch, weight_decay=1e-3, lr=2e-4)
task_prediction_record[0] = find_res(0, test_model)

# task 2
epoch=150
a, b, c = test_model.train_new_task(grouped_data_train[1],
                                    grouped_label_train[1], grouped_task_train[1], n_pseudo=128, lr=1e-4,
                                    batch_size=128, epoch=epoch, weight_decay=1e-3, lam=1.0)
task_prediction_record[1] = find_res(1, test_model)
print(task_prediction_record[1]['curr_error_rate'])

# task 3
epoch=150
a, b, c = test_model.train_new_task(grouped_data_train[2],
                                    grouped_label_train[2], grouped_task_train[2], n_pseudo=128, lr=1e-4,
                                    batch_size=64, epoch=epoch, weight_decay=1e-3, lam=1.0)
task_prediction_record[2] = find_res(2, test_model)
print(task_prediction_record[2]['curr_error_rate'])

# task 4
epoch=50
a, b, c = test_model.train_new_task(grouped_data_train[3],
                                    grouped_label_train[3], grouped_task_train[3], n_pseudo=128, lr=1e-4,
                                    batch_size=32, epoch=epoch, weight_decay=1e-3, lam=1.0)
task_prediction_record[3] = find_res(3, test_model)
print(task_prediction_record[3]['curr_error_rate'])

# task 5
epoch=100
a, b, c = test_model.train_new_task(grouped_data_train[4],
                                    grouped_label_train[4], grouped_task_train[4], n_pseudo=128, lr=1e-4,
                                    batch_size=32, epoch=epoch, weight_decay=1e-3, lam=2.0)
task_prediction_record[4] = find_res(4, test_model)
[_.mean(0) for _ in task_prediction_record[4]['curr_task_pred'].values()]
print(task_prediction_record[4]['curr_error_rate'])

# task 6
epoch=150  # 150, 16, 1e-4
a, b, c = test_model.train_new_task(grouped_data_train[5],
                                    grouped_label_train[5], grouped_task_train[5], n_pseudo=128, lr=1e-4,
                                    batch_size=32, epoch=epoch, weight_decay=1e-3, lam=2.0)
task_prediction_record[5] = find_res(5, test_model)
print(task_prediction_record[5]['curr_error_rate'])

# task 7
epoch=120
a, b, c = test_model.train_new_task(grouped_data_train[6],
                                    grouped_label_train[6], grouped_task_train[6], n_pseudo=128, lr=1e-4,
                                    batch_size=32, epoch=epoch, weight_decay=1e-3, lam=2.0)
task_prediction_record[6] = find_res(6, test_model)
plt.plot(range(epoch), a, label='test')
plt.plot(range(epoch), b, label='train')
plt.legend()
print(task_prediction_record[6]['curr_error_rate'])


# group specific task identity prediction
sample_info = pd.read_csv("iatlas-ici-sample_info.tsv", sep='\t').sample(frac=1., random_state=314159, ignore_index=True)
responder = sample_info['Responder']
grouped_responder_test = [torch.tensor(list(responder[ICI_task == i][:sum(ICI_task == i)//5]), dtype=torch.float) for i in ICI_task_look_up.keys()]
Atezo_pred = task_prediction_record[6]['curr_task_pred']['task0']
Nivo_pred = task_prediction_record[6]['curr_task_pred']['task1']

# Atezo vs None
prob_atezo = np.zeros((4, 7))
prob_atezo[1] = Atezo_pred[(grouped_label_test[0]==1)*(grouped_responder_test[0]==0)].mean(0)
prob_atezo[0] = Atezo_pred[(grouped_label_test[0]==0)*(grouped_responder_test[0]==0)].mean(0)
prob_atezo[3] = Atezo_pred[(grouped_label_test[0]==1)*(grouped_responder_test[0]==1)].mean(0)
prob_atezo[2] = Atezo_pred[(grouped_label_test[0]==0)*(grouped_responder_test[0]==1)].mean(0)


plt.imshow(prob_atezo)
plt.xticks(range(7), [r'$\bf{Atezo}$', 'Nivo', 'Pembro', 'None', 'Ipi', 'Ipi+\nPembro', 'Ipi+\nNivo'],fontsize=11)
plt.yticks(range(4), ['Non kidney, \n Non-responder', 'Kidney, \n Non-responder', 'Non kidney, \n Responder', 'Kidney, \n Responder'], fontsize=12)
plt.gca().get_yticklabels()[1].set_fontweight('bold')
plt.gca().get_yticklabels()[3].set_fontweight('bold')
plt.title('Avg. predicted probability, Task = Atezo', fontsize=15)
clb=plt.colorbar(fraction=0.03)
plt.xlabel('Candidate tasks', fontsize=12)
# clb.ax.tick_params(labelsize=12)
clb.set_label('Averaged predictive probability', rotation=270, labelpad=15)
plt.gcf().tight_layout()
plt.gcf().set_size_inches(6.5,4.5)
plt.savefig('Atezo_pred.png')
np.save('atezo_pred.npy', prob_atezo)


# Nivo vs Ipi
prob_nivo = np.zeros((4, 7))
prob_nivo[0] = Nivo_pred[(grouped_label_test[1]!=1)*(grouped_responder_test[1]==0)].mean(0)
prob_nivo[1] = Nivo_pred[(grouped_label_test[1]==1)*(grouped_responder_test[1]==0)].mean(0)
prob_nivo[2] = Nivo_pred[(grouped_label_test[1]!=1)*(grouped_responder_test[1]==1)].mean(0)
prob_nivo[3] = Nivo_pred[(grouped_label_test[1]==1)*(grouped_responder_test[1]==1)].mean(0)

plt.imshow(prob_nivo)
plt.xticks(range(7), ['Atezo', r'$\bf{Nivo}$', 'Pembro', 'None', 'Ipi', 'Ipi+\nPembro', 'Ipi+\nNivo'], fontsize=11)
plt.yticks(range(4), ['Non skin, \n Non-responder', 'Skin, \n Non-responder', 'Non skin, \n Responder', 'Skin, \nResponder'], fontsize=12)
plt.gca().get_yticklabels()[1].set_fontweight('bold')
plt.title('Avg. predicted probability, Task = Nivo', fontsize=15)
clb=plt.colorbar(fraction=0.03)
plt.xlabel('Candidate tasks', fontsize=12)
# clb.ax.tick_params(labelsize=12)
clb.set_label('Averaged predictive probability', rotation=270, labelpad=15)
plt.gcf().tight_layout()
plt.gcf().set_size_inches(6.5,4.5)
plt.savefig('Nivo_pred.png')
np.save('nivo_pred.npy', prob_nivo)


# knowledge retention
retention_first_3 = np.zeros((3, 7))
retention_first_3[0] = np.array([task_prediction_record[i]['curr_error_rate']['task{}'.format(0)] for i in range(0, 7)])
retention_first_3[1, 1:] = np.array([task_prediction_record[i]['curr_error_rate']['task{}'.format(1)] for i in range(1, 7)])
retention_first_3[2, 2:] = np.array([task_prediction_record[i]['curr_error_rate']['task{}'.format(2)] for i in range(2, 7)])
# np.save(file='ICI_retention.npy',
#         arr=retention_first_3)


# Overall task ID estimation
# np.save(file='ICI_heatmap.npy',
#         arr=torch.vstack([_.mean(0) for _ in task_prediction_record[6]['curr_task_pred'].values()]).numpy())
task_est = np.load('ICI_heatmap.npy')

plt.imshow(task_est)
clb=plt.colorbar()
clb.ax.tick_params(labelsize=12)
clb.set_label('Averaged predictive probability', fontsize=12, rotation=270, labelpad=30)
plt.xticks(ticks=range(7), labels=['Atezo', 'Nivo', 'Pembro', 'None', 'Ipi', 'Ipi+Pembro', 'Ipi+Nivo'])
plt.yticks(ticks=range(7), labels=['Atezo', 'Nivo', 'Pembro', 'None', 'Ipi', 'Ipi+Pembro', 'Ipi+Nivo'])
plt.gca().tick_params(axis='x', rotation=45, labelsize=12)
plt.gca().tick_params(axis='y', labelsize=12)
plt.title('Task identity estimation', fontsize=18)
plt.xlabel('Predicted task identity', fontsize=12)
plt.ylabel('True task identity', fontsize=12)
# plt.gcf().set_size_inches(17,15)
plt.gcf().tight_layout()
plt.savefig('task_id.png')
plt.close()


# Knowledge retention curve, note that we got classes with only one class, test error=0 by definition
ICI_retention = np.load('ICI_retention.npy')
ICI_retention = np.concatenate((ICI_retention, np.zeros((4,7))), 0)

# trail the oracle model
epoch = [150, 150, 150, 150, 150, 150, 150]
task_prediction_record_oracle = {}
for _ in range(7):
    my_train = torch.cat([grouped_data_train[i] for i in range(_+1)], 0)
    my_label = torch.cat([grouped_label_train[i] for i in range(_+1)])
    my_task = torch.cat([grouped_task_train[i] for i in range(_+1)])
    test_model = CLBruno(x_dim=ICI_data.shape[1], y_dim=256, task_dim=256, cond_dim=256,
                         task_num=_+1, y_cat_num=task_categories[:_+1], n_dense_block=6, n_hidden_dense=256,
                         activation=nn.Tanh(), mu_init=0., var_init=1., corr_init=0.1)

    train_loss, test_loss = test_model.train_init(my_train, my_label, my_task,
                                                  batch_size=128,
                                                  epoch=epoch[_], weight_decay=1e-3, lr=2e-4)
    task_prediction_record_oracle[_] = find_res(_, test_model)

import pickle
# filehandler = open('CL_ICI_oracle2.pkl', "wb")
# pickle.dump(task_prediction_record_oracle, filehandler)
# filehandler.close()




file = open("CL_ICI_oracle2.pkl",'rb')
task_prediction_record_oracle = pickle.load(file)
file.close()
retention_oracle = np.zeros((7, 7))
retention_oracle[0] = np.array([task_prediction_record_oracle[i]['curr_error_rate']['task{}'.format(0)] for i in range(0, 7)])
retention_oracle[1, 1:] = np.array([task_prediction_record_oracle[i]['curr_error_rate']['task{}'.format(1)] for i in range(1, 7)])
retention_oracle[2, 2:] = np.array([task_prediction_record_oracle[i]['curr_error_rate']['task{}'.format(2)] for i in range(2, 7)])


# plot retention curves
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
legend_elements = [Line2D([0], [0], marker='o', color='k', lw=2, label='TIL'),
                   Line2D([0], [0], marker='^', color='k', lw=2, label='Oracle', linestyle='dotted'),
                   Patch(facecolor='orchid', edgecolor='orchid', label='Atezo'),
                   Patch(facecolor='grey', edgecolor='grey', label='Nivo'),
                   Patch(facecolor='darkcyan', edgecolor='darkcyan', label='Pembro'),
                   Patch(facecolor='turquoise', edgecolor='turquoise', label='None'),
                   Patch(facecolor='dodgerblue', edgecolor='dodgerblue', label='Ipi'),
                   Patch(facecolor='darkviolet', edgecolor='darkviolet', label='Ipi+Pembro'),
                   Patch(facecolor='orange', edgecolor='orange', label='Ipi+Nivo')]

colors = ['orchid', 'grey', 'darkcyan', 'turquoise', 'dodgerblue', 'darkviolet', 'orange']
fig, ax = plt.subplots()
for i in range(7):
    loss = ICI_retention[i, i:]
    loss_oracle = retention_oracle[i, i:]
    ax.plot(range(i, 7), loss, marker='o', markersize=5, c=colors[i], alpha=0.7)
    ax.plot(range(i, 7), loss_oracle, marker='^', markersize=5, c=colors[i], alpha=0.6, linestyle='dotted')
ax.set_xlabel('Incremental step')
ax.set_ylabel('Misclassification rate')
ax.set_title('ICI data under TIL scenario')
ax.legend(handles=legend_elements, loc='upper left')
fig.tight_layout()
fig.set_size_inches(5.4,4.8)
plt.savefig('ICI_acc.png')

# np.save(file='ICI_retention.npy',
#         arr=ICI_retention)
# np.save(file='ICI_retention_oracle.npy',
#         arr=retention_oracle)





