import matplotlib.pyplot as plt
import torchvision
from CL_BRUNO import *


x_dim = 1000
n_class = [2, 3, 4, 5]
foci = [torch.tensor([[np.cos(2*np.pi*__/ncl), np.sin(2*np.pi*__/ncl)] for __ in range(ncl)], dtype=torch.float32) * np.sqrt((ncl-1)) for ncl in n_class]
n = 500
decay=1e-3
torch.manual_seed(31415)
np.random.seed(31415)
def foci_sampler(task, size):
    focus = foci[task]
    n_cluster = focus.shape[0]
    label = np.random.choice(range(n_cluster), size, p=np.ones(n_cluster)/n_cluster)
    return {'X': torch.randn(size, x_dim, dtype=torch.float32)*0.5 + focus[label].tile((1, x_dim//2)), 'y': torch.tensor(label), 'task': torch.tensor([task]*size)}


# model initialisation
test_model = CLBruno(x_dim=x_dim, y_dim=16, task_dim=16, cond_dim=32,
                     task_num=1, y_cat_num=[2], n_dense_block=6, n_hidden_dense=128, activation=nn.Tanh(),
                     mu_init=0., var_init=1., corr_init=0.1)

def find_res(q):
    error_rate = {}
    label_pred = {}
    task_pred = {}
    N=500
    datasets_test = [foci_sampler(t, N) for t in range(len(n_class))]
    for t in range(q+1):
        print(t)
        y_pred, t_pred = torch.zeros((N, t+2)), torch.zeros((N, q+1))
        for i in range(N):
            with torch.no_grad():
                temp, t_pred[i] = test_model.prediction(datasets_test[t]['X'][i])
            y_pred[i] = temp[t]
        my_pred = y_pred.argmax(1)
        task_pred['task{}'.format(t)] = t_pred
        error_rate['task{}'.format(t)] = torch.mean(1.0*(my_pred != datasets_test[t]['y'])).item()
        label_pred['task{}'.format(t)] = y_pred
    return {'curr_task_pred': task_pred, 'curr_error_rate': error_rate, 'curr_label_pred': label_pred}


# task 1, initialisation
datasets = [foci_sampler(t, n) for t in range(len(n_class))]
X_0 = datasets[0]['X']
y_0 = datasets[0]['y']
task_0 = datasets[0]['task']

train_error, test_error = test_model.train_init(X_0, y_0, task_0, batch_size=128, epoch=100, weight_decay=decay)
task_prediction_record = {}
task_prediction_record[0] = find_res(0)

# task 2, TIL
X_1 = datasets[1]['X']
y_1 = datasets[1]['y']
task_1 = datasets[1]['task']
a, b, c = test_model.train_new_task(X_1, y_1, task_1, batch_size=128, n_pseudo=128, epoch=60, weight_decay=decay, lam=1.)
task_prediction_record[1] = find_res(1)

# task 3, TIL
X_2 = datasets[2]['X']
y_2 = datasets[2]['y']
task_2 = datasets[2]['task']
a, b, c = test_model.train_new_task(X_2, y_2, task_2, batch_size=128, n_pseudo=128, epoch=70, weight_decay=decay, lam=1.)
task_prediction_record[2] = find_res(2)

# task 4, CIL, batch 1
my_id = [_ in [0,1,2] for _ in datasets[3]['y']]
X_3_1 = datasets[3]['X'][my_id]
y_3_1 = datasets[3]['y'][my_id]
task_3_1 = datasets[3]['task'][my_id]
a, b, c = test_model.train_new_task(X_3_1, y_3_1, task_3_1, batch_size=128, n_pseudo=128, epoch=150, weight_decay=decay, lam=1.)

# task 4, CIL, batch 2
X_3_2= datasets[3]['X'][~np.array(my_id)]
y_3_2 = datasets[3]['y'][~np.array(my_id)]
task_3_2 = datasets[3]['task'][~np.array(my_id)]
a, b, c = test_model.train_continual_task(X_3_2, y_3_2, task_id=3, batch_size=128, n_pseudo=128, epoch=200, weight_decay=decay, lam=1.)
task_prediction_record[3] = find_res(3)





# visualisation of the trained model

colll = ['orchid', 'grey', 'darkcyan', 'turquoise', 'dodgerblue', 'darkviolet']
fig, ax = plt.subplots(2, 2)
for i in range(4):
    a, b, c, d = test_model.task_specific_sampling(i, 400)
    for ccc in np.unique(b):
        my_id = ccc == b
        ax[i%2, i//2].scatter(c[my_id, 0].numpy(), c[my_id, 1].numpy(), c=colll[ccc.item()], marker='o', alpha=0.5)
        my_id_2 = ccc == datasets[i]['y'].numpy()
        ax[i%2, i//2].scatter(datasets[i]['X'][my_id_2, 0].numpy(), datasets[i]['X'][my_id_2, 1].numpy(),
                              c=colll[ccc.item()], alpha=0.5, marker='+')
    ax[i % 2, i // 2].set_title('Task {}'.format(i))
    ax[i % 2, i // 2].set_xlim([-3.5, 3.5])
    ax[i % 2, i // 2].set_ylim([-3.5, 3.5])
    ax[i % 2, i // 2].set_xlabel('Dim 1')
    ax[i % 2, i // 2].set_ylabel('Dim 2')
    ax[i % 2, i // 2].set_title('Task {}'.format(i+1), fontsize=12)

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='orchid', edgecolor='orchid', label='Class 1'),
                   Patch(facecolor='grey', edgecolor='grey', label='Class 2'),
                   Patch(facecolor='darkcyan', edgecolor='darkcyan', label='Class 3'),
                   Patch(facecolor='turquoise', edgecolor='turquoise', label='Class 4'),
                   Patch(facecolor='dodgerblue', edgecolor='dodgerblue', label='Class 5'),
                   Line2D([0], [0], marker='o', color='k', label='Generated', alpha=0.5,
                          markerfacecolor='k', markersize=10, linestyle='None',),
                   Line2D([0], [0], marker='+', color='k', label='Test set', alpha=0.5,
                          markerfacecolor='k', markersize=10, linestyle='None',)]
fig.legend(handles=legend_elements, loc='right', bbox_to_anchor=(1.01, 0.5))
# fig.tight_layout()
fig.set_size_inches(12, 10)
plt.savefig('gen_vs_test.png')
plt.close()


# heatmap of task-identity estimation
plt.imshow([_.mean(0).numpy() for _ in task_prediction_record[3]['curr_task_pred'].values()])
plt.xticks(range(4), labels=['task {}'.format(i+1) for i in range(4)])
plt.yticks(range(4), labels=['task {}'.format(i+1) for i in range(4)])

