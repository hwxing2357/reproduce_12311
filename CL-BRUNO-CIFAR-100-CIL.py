import matplotlib.pyplot as plt
import torch
import time
from CL_BRUNO_aistats import *
import pandas as pd
from torch.utils.data import Subset
import torchvision
import torchvision.transforms as transforms
device='cuda'


preprocess_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5070, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])
])

preprocess_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5070, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])
])


cifar_train = torchvision.datasets.CIFAR100(root='./CIFAR100', train=True, transform=preprocess_train)
cifar_test = torchvision.datasets.CIFAR100(root='./CIFAR100', train=False, transform=preprocess_test)

def get_subset(list_of_indices, dataset):
    mask = torch.tensor([False]*len(dataset))
    for i in list_of_indices:
        idx = torch.tensor(dataset.targets) == i
        mask = mask | idx
    mask = mask.nonzero().reshape(-1)
    return Subset(dataset, mask)


# pretrain feature vector using the first batch of 10 classes
cifar_01_train = get_subset(list(range(10)), cifar_train)
cifar_01_test = get_subset(list(range(10)), cifar_test)
cifar_01_train_loader = DataLoader(cifar_01_train, batch_size=64)
cifar_01_test_loader = DataLoader(cifar_01_test, batch_size=64)
dataset_sizes = {'train': len(cifar_01_train), 'val': len(cifar_01_test)}
my_loader = {'train': cifar_01_train_loader, 'val': cifar_01_test_loader}


# define the resnet18 based feature extractor
def create_model(init_out=2):
    model = torchvision.models.resnet18(weights='DEFAULT')
    for param in model.parameters():
        if len(param.shape) != 1:
            param.requires_grad = False
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
    model.layer4[1].relu = nn.Tanh()
    # model.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.maxpool = nn.Identity()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, init_out)
    return model


model_ft = create_model(init_out=10).to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
# optimizer_ft = torch.optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
optimizer_ft = torch.optim.Adam(model_ft.parameters(), lr=1e-3)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=50, gamma=0.1)



def train_model(model, criterion, optimizer, scheduler, dataloaders, num_epochs=20):
    since = time.time()
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            print()
        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    return model


model_fted = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, my_loader, num_epochs=25)
model_fted = model_fted.eval()
model_fted.fc = nn.Identity()






cifar_train = torchvision.datasets.CIFAR100(root='./CIFAR100', train=True, transform=preprocess_test)
cifar_test = torchvision.datasets.CIFAR100(root='./CIFAR100', train=False, transform=preprocess_test)
cifar_loader_train = DataLoader(cifar_train, batch_size=64)
cifar_loader_test = DataLoader(cifar_test, batch_size=64)

with torch.no_grad():
    transformed_x_train = []
    transformed_y_train = torch.tensor([])
    for x, y in cifar_loader_train:
        transformed_x_train += [model_fted(x.to(device))]
        transformed_y_train = torch.cat((transformed_y_train, y))
    transformed_x_train = torch.vstack(transformed_x_train).cpu()
    transformed_y_train = transformed_y_train.to(torch.long)

    transformed_x_test = []
    transformed_y_test = torch.tensor([])
    for x, y in cifar_loader_test:
        transformed_x_test += [model_fted(x.to(device))]
        transformed_y_test = torch.cat((transformed_y_test, y))
    transformed_x_test = torch.vstack(transformed_x_test).cpu()
    transformed_y_test = transformed_y_test.to(torch.long)


torch.save(model_fted.state_dict(), 'mnist_examplar_feature_extractor.pt')
pd.DataFrame(transformed_x_train.numpy()).to_csv('CIFAR100_pretrained_feature_train.csv', index=False)
pd.DataFrame(transformed_x_test.numpy()).to_csv('CIFAR100_pretrained_feature_test.csv', index=False)



import umap
a = umap.UMAP(random_state=42)
a.fit(transformed_x_test.numpy())
plt.scatter(a.embedding_[:, 0], a.embedding_[:, 1], c=transformed_y_test.numpy())
plt.savefig('umap_cifar100.png')
plt.close()









preprocess_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5070, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])
])

preprocess_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5070, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])
])
cifar_train = torchvision.datasets.CIFAR100(root='./CIFAR100', train=True, transform=preprocess_train)
cifar_test = torchvision.datasets.CIFAR100(root='./CIFAR100', train=False, transform=preprocess_test)
transformed_y_test = torch.tensor(cifar_test.targets, dtype=torch.long)
transformed_y_train = torch.tensor(cifar_train.targets, dtype=torch.long)
transformed_x_train = torch.tensor(pd.read_csv('CIFAR100_pretrained_feature_train.csv').to_numpy(), dtype=torch.float)
transformed_x_test = torch.tensor(pd.read_csv('CIFAR100_pretrained_feature_test.csv').to_numpy(), dtype=torch.float)


# task_split = [[0, 1, 2, 3], [4, 5], [6, 7], [8, 9]]
task_split = [list(range(i*10, (i+1)*10)) for i in range(10)]
CIL_cifar_train = {}
CIL_cifar_test = {}
# nhwc to nchw
for _ in range(len(task_split)):
    task_id = np.array([i for i,j in enumerate(transformed_y_train) if j in task_split[_]])
    CIL_cifar_train['X_{}'.format(_)] = transformed_x_train[task_id]
    CIL_cifar_train['y_{}'.format(_)] = transformed_y_train[task_id]
    task_id = np.array([i for i, j in enumerate(transformed_y_test) if j in task_split[_]])
    CIL_cifar_test['X_{}'.format(_)] = transformed_x_test[task_id]
    CIL_cifar_test['y_{}'.format(_)] = transformed_y_test[task_id]


x_dim = CIL_cifar_train['X_0'].shape[1]

# initialise the model
test_model = CLBruno(x_dim=x_dim, y_dim=128, task_dim=1, cond_dim=129, conv=False, task_num=1,
                     y_cat_num=[10], single_task=True, n_dense_block=6, n_hidden_dense=128,
                     activation=nn.Tanh(), mu_init=0., var_init=1., corr_init=0.1)


test_model = test_model.to('cuda')
train_loss, test_loss = test_model.train_init(CIL_cifar_train['X_0'], CIL_cifar_train['y_0'],
                                              torch.zeros(CIL_cifar_train['y_0'].shape[0], dtype=torch.long),
                                              batch_size=128, epoch=30, weight_decay=0., lr=1e-3, embedding_reg=0.1,
                                              context_portion=0.2)


N = len(CIL_cifar_test['y_0'])
my_id_test = range(len(CIL_cifar_test['y_0']))
# runnable, check outputs
q = torch.zeros((N, 10))
p = torch.zeros(N)
for i,j in enumerate(my_id_test):
    a, b = test_model.prediction(CIL_cifar_test['X_0'][j], 0)
    q[i] = a.cpu()
    p[i] = b.cpu()
print(torch.sum(q.cpu().argmax(1) != CIL_cifar_test['y_0'][my_id_test])/N)

batch_sizes = [128]*9
# doing CIL
start = time.time()
for batch_id in range(1, 10):
    train_loss1, test_loss1, reg_loss1 = test_model.train_continual_task(X_new=CIL_cifar_train['X_{}'.format(batch_id)],
                                                                         y_new=CIL_cifar_train['y_{}'.format(batch_id)],
                                                                         task_id=0, epoch=30, batch_size=int(batch_sizes[batch_id-1]),
                                                                         weight_decay=0., lr=1e-3, n_pseudo=128,
                                                                         embedding_reg=0.1)

    acc = 0.
    for hist_id in range(batch_id+1):
        N = len(CIL_cifar_test['y_{}'.format(hist_id)])
        my_id_test = range(len(CIL_cifar_test['y_{}'.format(hist_id)]))
        q = torch.zeros((N, (batch_id + 1) * 10))
        p = torch.zeros(N)
        for i, j in enumerate(my_id_test):
            a, b = test_model.prediction(CIL_cifar_test['X_{}'.format(hist_id)][j], 0)
            q[i] = a.cpu()
            p[i] = b.cpu()
        print(torch.sum(q.cpu().argmax(1) != CIL_cifar_test['y_{}'.format(hist_id)][my_id_test]) / N)
        acc += torch.sum(q.cpu().argmax(1) != CIL_cifar_test['y_{}'.format(hist_id)][my_id_test]) / N
    print('Batch {}'.format(batch_id))
    print(acc/(batch_id+1))
print(time.time() - start)


torch.save(test_model.state_dict(), 'CIFAR100_L6_testmodel_30_n128.pt')
