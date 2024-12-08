import torch

torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from PIL import Image
from tqdm import tqdm
import time
import copy
import argparse
import torchvision.models as models
import torchvision.transforms as transforms
import random
import pandas as pd

class Arg:
    def __init__(self, outfile='temp_LwF_PAN.csv', matr='acc_matr_LwF_PAN.npz', num_classes=32,
                 init_lr=1e-4, num_epochs=100, batch_size=128):
        self.outfile = outfile
        self.matr = matr
        self.num_classes = num_classes
        self.init_lr = init_lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size

args = Arg()

# code taken from https://github.com/ngailapdi/LWF

def MultiClassCrossEntropy(logits, labels, T):
    # Ld = -1/N * sum(N) sum(C) softmax(label) * log(softmax(logit))
    labels = Variable(labels.data, requires_grad=False)
    outputs = torch.log_softmax(logits / T, dim=1)  # compute the log of softmax values
    labels = torch.softmax(labels / T, dim=1)
    # print('outputs: ', outputs)
    # print('labels: ', labels.shape)
    outputs = torch.sum(outputs * labels, dim=1, keepdim=False)
    outputs = -torch.mean(outputs, dim=0, keepdim=False)
    # print('OUT: ', outputs)
    return Variable(outputs.data, requires_grad=True)

class MLP(nn.Module):
    def __init__(self, n_dim=2000, hidden_size=400, out=10):
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

def kaiming_normal_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='sigmoid')


class Model(nn.Module):
    def __init__(self, classes, classes_map, args):
        # Hyper Parameters
        self.init_lr = args.init_lr
        self.num_epochs = args.num_epochs
        self.batch_size = args.batch_size
        self.lower_rate_epoch = [int(0.7 * self.num_epochs), int(0.9 * self.num_epochs)]  # hardcoded decay schedule
        self.lr_dec_factor = 10

        self.pretrained = False
        self.momentum = 0.9
        self.weight_decay = 0.0001
        # Constant to provide numerical stability while normalizing
        self.epsilon = 1e-16

        # Network architecture
        super(Model, self).__init__()
        self.model = MLP(2000, 400, classes)
        # self.model = models.resnet34(pretrained=self.pretrained)
        self.model.apply(kaiming_normal_init)

        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, classes, bias=False)
        self.fc = self.model.fc
        self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])
        # self.feature_extractor = nn.DataParallel(self.feature_extractor)

        # n_classes is incremented before processing new data in an iteration
        # n_known is set to n_classes after all data for an iteration has been processed
        self.n_classes = 0
        self.n_known = 0
        self.classes_map = classes_map

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def increment_classes(self, new_classes):
        """Add n classes in the final fc layer"""
        n = len(new_classes)
        print('new classes: ', n)
        in_features = self.fc.in_features
        out_features = self.fc.out_features
        weight = self.fc.weight.data

        if self.n_known == 0:
            new_out_features = n
        else:
            new_out_features = out_features + n
        print('new out features: ', new_out_features)
        self.model.fc = nn.Linear(in_features, new_out_features, bias=False)
        self.fc = self.model.fc

        kaiming_normal_init(self.fc.weight)
        self.fc.weight.data[:out_features] = weight
        self.n_classes += n

    def classify(self, images):
        """Classify images by softmax

        Args:
            x: input image batch
        Returns:
            preds: Tensor of size (batch_size,)
        """
        _, preds = torch.max(torch.softmax(self.forward(images), dim=1), dim=1, keepdim=False)

        return preds

    def update(self, dataset, class_map, args):

        self.compute_means = True

        # Save a copy to compute distillation outputs
        prev_model = copy.deepcopy(self)
        # prev_model.cuda()

        classes = list(set(dataset.train_labels))
        # print("Classes: ", classes)
        print('Known: ', self.n_known)
        if self.n_classes == 1 and self.n_known == 0:
            new_classes = [classes[i] for i in range(1, len(classes))]
        else:
            new_classes = [cl for cl in classes if class_map[cl] >= self.n_known]

        if len(new_classes) > 0:
            self.increment_classes(new_classes)
            # self.cuda()

        loader = torch.utils.data.DataLoader(dataset, batch_size=128)

        print("Batch Size (for n_classes classes) : ", len(dataset))
        optimizer = optim.SGD(self.parameters(), lr=self.init_lr, momentum=self.momentum,
                              weight_decay=self.weight_decay)

        with tqdm(total=self.num_epochs) as pbar:
            for epoch in range(self.num_epochs):

                # Modify learning rate
                # if (epoch+1) in lower_rate_epoch:
                # 	self.lr = self.lr * 1.0/lr_dec_factor
                # 	for param_group in optimizer.param_groups:
                # 		param_group['lr'] = self.lr

                for i, (indices, images, labels) in enumerate(loader):
                    seen_labels = []
                    images = Variable(torch.FloatTensor(images))
                    seen_labels = torch.LongTensor([class_map[label] for label in labels.numpy()])
                    labels = Variable(seen_labels)
                    # indices = indices

                    optimizer.zero_grad()
                    logits = self.forward(images)
                    cls_loss = nn.CrossEntropyLoss()(logits, labels)
                    if self.n_classes // len(new_classes) > 1:
                        dist_target = prev_model.forward(images)
                        logits_dist = logits[:, :-(self.n_classes - self.n_known)]
                        dist_loss = MultiClassCrossEntropy(logits_dist, dist_target, 2)
                        loss = dist_loss + cls_loss
                    else:
                        loss = cls_loss

                    loss.backward()
                    optimizer.step()

                    if (i + 1) % 1 == 0:
                        tqdm.write('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                                   % (epoch + 1, self.num_epochs, i + 1, np.ceil(len(dataset) / self.batch_size),
                                      loss.data))

                pbar.update(1)



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
num_task=6
PAN_CAN_data_train, PAN_CAN_data_test = {}, {}
PAN_CAN_label_train, PAN_CAN_label_test = {}, {}

for i, idx in enumerate(label_grouping):
    my_index = np.array([_ in idx for _ in PAN_CAN_label_s])
    n = sum(my_index)
    partition = np.random.permutation(range(n))
    PAN_CAN_data_test[i] = torch.tensor(PAN_CAN_s[my_index][partition[:n//5]], dtype=torch.float)
    PAN_CAN_label_test[i] = PAN_CAN_label_s[my_index][partition[:n//5]]
    PAN_CAN_data_train[i] = torch.tensor(PAN_CAN_s[my_index][partition[n//5:]], dtype=torch.float)
    PAN_CAN_label_train[i] = PAN_CAN_label_s[my_index][partition[(n // 5):]]

task_sample_size = [PAN_CAN_data_test[i].shape[0] for i in range(num_task)]

class MyDataset(torch.utils.data.Dataset):
    # merge all datasets, use task to identify different tasks
    def __init__(self, X, y):
        self.train_data = X
        self.train_labels = y
    def __len__(self):
        return self.train_data.shape[0]

    def __getitem__(self, idx):
        # vec, int, int
        return idx, self.train_data[idx], self.train_labels[idx]

my_train_loader, my_test_loader = {}, {}
for i in range(num_task):
    my_train_loader[i] = torch.utils.data.DataLoader(MyDataset(PAN_CAN_data_train[i], PAN_CAN_label_train[i]), batch_size=128)
    my_test_loader[i] = torch.utils.data.DataLoader(MyDataset(PAN_CAN_data_test[i], PAN_CAN_label_test[i]), batch_size=128)

my_train_set, my_test_set = {}, {}
for i in range(num_task):
    my_train_set[i] = MyDataset(PAN_CAN_data_train[i], PAN_CAN_label_train[i])
    my_test_set[i] = MyDataset(PAN_CAN_data_test[i], PAN_CAN_label_test[i])



num_classes = args.num_classes

total_classes = 32
all_classes = np.arange(total_classes)

n_cl_temp = 0
num_iters = total_classes//num_classes
class_map = {}
map_reverse = {}
for i, cl in enumerate(all_classes):
    if cl not in class_map:
        class_map[cl] = int(n_cl_temp)
        n_cl_temp += 1

print ("Class map:", class_map)

for cl, map_cl in class_map.items():
    map_reverse[map_cl] = int(cl)

print ("Map Reverse:", map_reverse)

print ("all_classes:", all_classes)


with open(args.outfile, 'w') as file:
    print("Classes, Train Accuracy, Test Accuracy", file=file)
    model = Model(1, class_map, args)
    acc_matr = np.zeros(num_task)
    for s in range(num_task):
        print('Iteration: ', s)
        print("Loading training examples for classes", all_classes[s: s+num_classes])
        train_set = my_train_set[s]
        train_loader = my_train_loader[s]
        test_loader = my_test_loader[s]
        test_set = my_test_set[s]
        model.update(train_set, class_map, args)
        model.eval()
        model.n_known = model.n_classes
        print("%d, " % model.n_known, file=file, end="")
        print("model classes : %d, " % model.n_known)
        total = 0.0
        correct = 0.0
        for indices, images, labels in train_loader:
            images = Variable(images)
            preds = model.classify(images)
            preds = [map_reverse[pred] for pred in preds.cpu().numpy()]
            total += labels.size(0)
            correct += (preds == labels.numpy()).sum()

        print ('%.2f ,' % (100.0 * correct / total), file=file, end="")
        print ('Train Accuracy : %.2f ,' % (100.0 * correct / total))

        total = 0.0
        correct = 0.0
        for indices, images, labels in test_loader:
            images = Variable(images)
            preds = model.classify(images)
            preds = [map_reverse[pred] for pred in preds.cpu().numpy()]
            total += labels.size(0)
            correct += (preds == labels.numpy()).sum()

        print ('%.2f' % (100.0 * correct / total), file=file)
        print ('Test Accuracy : %.2f' % (100.0 * correct / total))


    for s in range(num_task):
        test_set = my_test_set[s]
        test_loader = my_test_loader[s]
        total = 0.0
        correct = 0.0
        for indices, images, labels in test_loader:
            images = Variable(images)
            preds = model.classify(images)
            preds = [map_reverse[pred] for pred in preds.cpu().numpy()]
            total += labels.size(0)
            correct += (preds == labels.numpy()).sum()
        acc_matr[s] = (100 * correct / total)
        print("Accuracy matrix", acc_matr[s])
        model.train()
    np.save(file=args.matr, arr=acc_matr)

