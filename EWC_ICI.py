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

epochs = 50
lr = 1e-3
batch_size = 64
sample_size = 200
hidden_size = 200
num_task = 7
n_dim = 500
pred_num = 5


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
task_sample_size = [sum(ICI_task == i) for i in range(7)]
ICI_task_label = ICI_data['ICI_Rx']
ICI_tissue_label = ICI_data['Cancer_tissue']
ICI_data = ICI_data.drop(['ICI_Rx', 'Cancer_tissue'], axis=1).to_numpy()
# ICI_data += 1e-2*np.random.randn(ICI_data.shape[0], ICI_data.shape[1])
ICI_label = pd.factorize(ICI_tissue_label)[0]
ICI_label_lookup = pd.factorize(ICI_tissue_label)[1]

# ICI_data = umap.UMAP(random_state=314159, n_components=200).fit(ICI_data).embedding_
grouped_task_test = [torch.tensor(ICI_task[ICI_task == i][:sum(ICI_task == i)//5], dtype=torch.long) for i in ICI_task_look_up.keys()]
grouped_task_train = [torch.tensor(ICI_task[ICI_task == i][sum(ICI_task == i)//5:], dtype=torch.long) for i in ICI_task_look_up.keys()]
grouped_label_test = [torch.tensor(ICI_label[ICI_task == i][:sum(ICI_task == i)//5], dtype=torch.long) for i in ICI_task_look_up.keys()]
grouped_label_train = [torch.tensor(ICI_label[ICI_task == i][sum(ICI_task == i)//5:], dtype=torch.long) for i in ICI_task_look_up.keys()]
grouped_data_test = [torch.tensor(ICI_data[ICI_task == i][:sum(ICI_task == i)//5], dtype=torch.float) for i in ICI_task_look_up.keys()]
grouped_data_train = [torch.tensor(ICI_data[ICI_task == i][sum(ICI_task == i)//5:], dtype=torch.float) for i in ICI_task_look_up.keys()]


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



for i in range(7):
    train_loader[i] = torch.utils.data.DataLoader(MyDataset(grouped_data_train[i], grouped_label_train[i]), batch_size=batch_size)
    test_loader[i] = torch.utils.data.DataLoader(MyDataset(grouped_data_test[i], grouped_label_test[i]), batch_size=batch_size)




class MLP(nn.Module):
    def __init__(self, hidden_size=400):
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

print(np.sum([task_sample_size[i]*acc_ewc[i][-1].item() for i in range(7)])/np.sum(task_sample_size))





