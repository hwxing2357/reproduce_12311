import os
import numpy as np
import dill
import copy
import time
import random
import pdb
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.nn import parameter
import torchvision.models as models

# taken from https://www.kaggle.com/code/iceicyflagflogfrog/lwf-using-cifar100-to-10-task-version-1-0
def fix_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

DEVICE='cpu'
class MLP(nn.Module):
    def __init__(self, n_dim=500, hidden_size=400, out=5):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, out)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.fc(x)

# LwF
class LwF(nn.Module):
    def __init__(self, h_arg, classes=5):
        self.classifier_LwF = {}
        self.lr = h_arg['lr']
        self.EPOCHS = h_arg['epochs']
        self.Batch_size = h_arg['batch_size']
        self.momentum = h_arg['momentum']
        self.weight_decay = h_arg['weight_decay']
        self.classes = classes
        self.pretrained = True
        # --------------------------------------------------------------------------

        super(LwF, self).__init__()


        self.feature_extractor = MLP()
        self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-1])
        self.feature_extractor = self.feature_extractor.to(DEVICE)

        # task 추가 할때 마다 딕셔너리에 저장하기위해서 하나만 만들어 놓자
        # self.classifier = nn.Linear(models.resnet18().fc.in_features,self.classes)

    # --------------------------------------------------------------------------
    def forward(self, x):
        # with torch.autograd.set_detect_anomaly(True):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        # output = self.fc(x)
        outputs = {}
        # newtask를 위한 classifier도 같이 저장이 된다!!

        for t in self.classifier_LwF:
            self.fc = copy.deepcopy(self.classifier_LwF[t])
            outputs[t] = self.fc(x)

        return outputs

    def make_response(self, train_loader, response, task):
        print('make response')
        for t in self.classifier_LwF:
            if t != task:
                response[t] = []

        self.feature_extractor.train(False)
        for t in self.classifier_LwF:
            self.classifier_LwF[t].train(False)

        for images, labels in train_loader:
            images = images.to(DEVICE)

            outputs = self.forward(images)
            # print('make response:',outputs['task1'].shape)

            for t in self.classifier_LwF:
                if t != task:
                    # output값을 리스트에 전체를! 리스트에 저장해서 넘겨주고
                    # 트레인에서는 idx값을 받아서 리스트에서 회정하면 계산
                    # 배치사이즈가 32일떄 마지막 배치는 8개밖에 없어서 차원오류가 발생!!!
                    # out_list.append(outputs[t].cpu())
                    response[t].append(outputs[t].cpu().detach())

        return response

    def LwF_criterion(self, logits, target, T):
        target.requires_grad = False
        #         target = target.detach()
        # print('LwF_criterion')
        # print(logits.shape)
        # print(target.shape)
        old_loss = torch.softmax(logits / T, dim=1) * torch.log_softmax(target / T, dim=1)
        old_loss = torch.sum(old_loss, dim=1, keepdim=False)
        old_loss = -torch.mean(old_loss, dim=0, keepdim=False)

        return old_loss

    def train_model(self, criterion, train_loader, test_loader, task):
        if task not in self.classifier_LwF:
            print('make classifier in classifier_LwF!!')
            self.classifier_LwF[task] = nn.Linear(400, self.classes)
            self.classifier_LwF[task] = self.classifier_LwF[task].to(DEVICE)
        print(self.classifier_LwF)

        if len(self.classifier_LwF) > 1:
            params = [
                {'params': self.feature_extractor.parameters(), 'lr': 0.02 * self.lr}
            ]
            for i in self.classifier_LwF:
                if i != task:
                    params = params + [{'params': self.classifier_LwF[i].parameters(), 'lr': 0.02 * self.lr}]
                else:
                    params = params + [{'params': self.classifier_LwF[i].parameters()}]
        else:
            print('first task!')
            params = [
                {'params': self.feature_extractor.parameters()},
                {'params': self.classifier_LwF[task].parameters()}
            ]
        print('param:', params)
        optimizer = torch.optim.SGD(params, lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)

        data_loader = {}
        data_loader['train'] = train_loader
        data_loader['val'] = test_loader

        # response를 딕셔너리 형태로 구현해서 가지고 있자(new classifier X)
        response = {}
        response = self.make_response(train_loader, response, task)
        print('len of response:', len(response))
        # balance parameter between new and old
        balance_wts = 1.

        since = time.time()
        best_model_wts = copy.deepcopy(self.state_dict())
        best_acc = 0.

        for epoch in range(self.EPOCHS):
            if epoch % 30 == 0:
                print('\t[EPOCHS{}/{}]'.format(epoch + 1, self.EPOCHS))
                print('-' * 50)
            for phase in ['train', 'val']:
                if phase == 'train':
                    if epoch % 30 == 0:
                        print('=================train phase=================')
                    self.feature_extractor.train()
                    for t in self.classifier_LwF:
                        self.classifier_LwF[t].train()
                #                     self.fc.train()
                else:
                    if epoch % 30 == 0:
                        print('=================val phase=================')
                    self.feature_extractor.eval()
                    for t in self.classifier_LwF:
                        self.classifier_LwF[t].eval()
                #                     self.fc.eval()

                running_loss = 0.
                running_corrects = 0.
                #                 total_loss = 0.
                for idx, (images, labels) in enumerate(data_loader[phase]):
                    images = images.to(DEVICE)
                    labels = labels.to(DEVICE)
                    optimizer.zero_grad()

                    outputs = self.forward(images)
                    _, preds = torch.max(outputs[task].data, 1)

                    if phase == 'train' and len(self.classifier_LwF) > 1:
                        total_loss = 0.
                        for i in self.classifier_LwF:
                            if i != task:
                                response[i][idx] = response[i][idx].to(DEVICE)
                                total_loss = total_loss + self.LwF_criterion(outputs[i], response[i][idx], 2)

                        loss = criterion(outputs[task], labels) + balance_wts * total_loss
                    else:
                        loss = criterion(outputs[task], labels)

                    if phase == 'train':
                        # with torch.autograd.detect_anomaly():
                        loss.backward()
                        optimizer.step()
                    running_loss += loss.item()
                    running_corrects += torch.sum(preds == labels.data).item()
                epoch_loss = running_loss / len(data_loader[phase].dataset)
                epoch_acc = 100. * running_corrects / len(data_loader[phase].dataset)

                if epoch % 30 == 0:
                    print('[phase:{}] Loss : {:.4f} Acc: {:.2f}%'.format( \
                        phase, epoch_loss, epoch_acc))

                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.state_dict())

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:.2f}%\n'.format(best_acc))

        self.load_state_dict(best_model_wts)

    ############################################################################################################

    def test_model(self, criterion, test_loader, task):
        for t in self.classifier_LwF:
            self.classifier_LwF[t].eval()
        self.feature_extractor.eval()

        running_loss = 0.
        running_corrects = 0.

        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = self.forward(images)
            _, preds = torch.max(outputs[task].data, 1)

            loss = criterion(outputs[task], labels)

            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels.data).item()
        test_loss = running_loss / len(test_loader.dataset)
        test_acc = 100. * running_corrects / len(test_loader.dataset)
        print('Test loss : {:.4f}\t Test Acc : {:.2f}%'.format(test_loss, test_acc))
        return test_acc


#############################################################################################
#                   MAIN

fix_seed(777)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

arg_dict = {
    'lr': 0.003,
    'epochs': 100,
    'batch_size': 128,
    'momentum': 0.9,
    'weight_decay': 0.0005
}



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

train_loader, test_loader = {}, {}
print(DEVICE)
# 경로 재설정 필요

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
    train_loader[i] = torch.utils.data.DataLoader(MyDataset(grouped_data_train[i], grouped_label_train[i]), batch_size=64)
    test_loader[i] = torch.utils.data.DataLoader(MyDataset(grouped_data_test[i], grouped_label_test[i]), batch_size=64)



n_classes = 5
total_task = len(train_loader)
print('the number of total_task:', len(train_loader))

model = LwF(arg_dict, classes=n_classes)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
#   TRAIN
for idx in range(total_task):
    train_D = train_loader[idx]
    test_D = test_loader[idx]
    task = 'task' + str(idx + 1)
    print('-' * 80)
    print('\t', task)
    model.train_model(criterion=criterion, train_loader=train_D, test_loader=test_D, task=task)
    print('-' * 80)
    print()
#   TEST
test_acc = [0 for i in range(total_task)]
for idx in range(total_task):
    test_D = test_loader[idx]
    task = 'task' + str(idx + 1)
    print('-' * 80)
    print(task)
    test_acc[idx] = model.test_model(criterion, test_D, task)
    print('-' * 80)
    print()

[(1-test_acc[i]/100)*len(test_loader[i])/sum([len(test_loader[j]) for j in range(7)]) for i in range(7)]
