from copy import deepcopy
import random
from torchvision import datasets
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# code taken from https://github.com/moskomule/ewc.pytorch/blob/master/demo.ipynb
use_cuda = False


def variable(t: torch.Tensor, use_cuda=use_cuda, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)


class EWC(object):
    def __init__(self, model: nn.Module, dataset: list):

        self.model = model
        self.dataset = dataset

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()

        for n, p in deepcopy(self.params).items():
            self._means[n] = variable(p.data)

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = variable(p.data)

        self.model.eval()
        for input in self.dataset:
            self.model.zero_grad()
            input = variable(input)
            output = self.model(input).view(1, -1)
            label = output.max(1)[1].view(-1)
            loss = F.nll_loss(F.log_softmax(output, dim=1), label)
            loss.backward()

            for n, p in self.model.named_parameters():
                precision_matrices[n].data += p.grad.data ** 2 / len(self.dataset)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss


def normal_train(model: nn.Module, optimizer: torch.optim, data_loader: torch.utils.data.DataLoader):
    model.train()
    epoch_loss = 0
    for input, target in data_loader:
        input, target = variable(input), variable(target)
        optimizer.zero_grad()
        output = model(input)
        loss = F.cross_entropy(output, target)
        epoch_loss += loss.detach().item()
        loss.backward()
        optimizer.step()
    return epoch_loss / len(data_loader)


def ewc_train(model: nn.Module, optimizer: torch.optim, data_loader: torch.utils.data.DataLoader,
              ewc: EWC, importance: float):
    model.train()
    epoch_loss = 0
    for input, target in data_loader:
        input, target = variable(input), variable(target)
        optimizer.zero_grad()
        output = model(input)
        loss = F.cross_entropy(output, target) + importance * ewc.penalty(model)
        epoch_loss += loss.detach().item()
        loss.backward()
        optimizer.step()
    return epoch_loss / len(data_loader)


def test(model: nn.Module, data_loader: torch.utils.data.DataLoader):
    model.eval()
    correct = 0
    for input, target in data_loader:
        input, target = variable(input), variable(target)
        output = model(input)
        correct += torch.sum(F.softmax(output, dim=1).max(dim=1)[1] == target)
    return correct / len(data_loader.dataset)

# data = dictionary of dataloaders, each returns x and y associated with tasks
train_loader, test_loader = {}, {}

epochs = 300
lr = 1e-4
batch_size = 128
sample_size = 200
hidden_size = 400
num_task = 6
n_dim = 2000
pred_num = 32


torch.manual_seed(31415)
np.random.seed(314159)
PAN_CAN = pd.read_csv('EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena', sep='\t')
PAN_CAN_meta = pd.read_csv('Survival_SupplementalTable_S1_20171025_xena_sp', sep='\t')

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

task_sample_size = [PAN_CAN_data_test[i].shape[0] for i in range(num_task)]

class MyDataset(torch.utils.data.Dataset):
    # merge all datasets, use task to identify different tasks
    def __init__(self, X, y):
        self.train_data = X
        self.train_label = y
    def __len__(self):
        return self.train_data.shape[0]

    def __getitem__(self, idx):
        # vec, int, int
        return self.train_data[idx], self.train_label[idx]

    def get_sample(self, sample_size):
        sample_idx = random.sample(range(len(self)), min(sample_size, len(self)))
        return [img for img in self.train_data[sample_idx]]



for i in range(num_task):
    train_loader[i] = torch.utils.data.DataLoader(MyDataset(PAN_CAN_data_train[i], PAN_CAN_label_train[i]), batch_size=batch_size)
    test_loader[i] = torch.utils.data.DataLoader(MyDataset(PAN_CAN_data_test[i], PAN_CAN_label_test[i]), batch_size=batch_size)




class MLP(nn.Module):
    def __init__(self, hidden_size=hidden_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, pred_num)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.fc5(x)


def ewc_process(epochs, importance, use_cuda=use_cuda, weight=None):
    model = MLP(hidden_size)
    if torch.cuda.is_available() and use_cuda:
        model.cuda()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=lr)

    loss, acc, ewc = {}, {}, {}
    for task in range(num_task):
        loss[task] = []
        acc[task] = []

        if task == 0:
            if weight:
                model.load_state_dict(weight)
            else:
                for _ in tqdm(range(epochs)):
                    loss[task].append(normal_train(model, optimizer, train_loader[task]))
                    acc[task].append(test(model, test_loader[task]))
        else:
            old_tasks = []
            for sub_task in range(task):
                old_tasks = old_tasks + train_loader[sub_task].dataset.get_sample(sample_size)
            old_tasks = random.sample(old_tasks, k=sample_size)
            for _ in tqdm(range(epochs)):
                loss[task].append(ewc_train(model, optimizer, train_loader[task], EWC(model, old_tasks), importance))
                for sub_task in range(task + 1):
                    acc[sub_task].append(test(model, test_loader[sub_task]))

    return loss, acc

def loss_plot(x):
    for t, v in x.items():
        plt.plot(list(range(t * epochs, (t + 1) * epochs)), v)

def accuracy_plot(x):
    for t, v in x.items():
        plt.plot(list(range(t * epochs, num_task * epochs)), v)
    plt.ylim(0, 1)


loss_ewc, acc_ewc = ewc_process(epochs, importance=1000)
loss_plot(loss_ewc)
accuracy_plot(acc_ewc)

print(np.sum([task_sample_size[i]*acc_ewc[i][-1].item() for i in range(num_task)])/np.sum(task_sample_size))


[0.4521, 0.3977, 0.4728, 0.4231, 0.4549]


