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
    transforms.Normalize(mean=[0.1307], std=[0.3081])
])

preprocess_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1307], std=[0.3081])
])


mnist_train = torchvision.datasets.MNIST(root='./MNIST', train=True, transform=preprocess_train, download=True)
mnist_test = torchvision.datasets.MNIST(root='./MNIST', train=False, transform=preprocess_test, download=True)




def get_subset(list_of_indices, dataset):
    mask = torch.tensor([False]*len(dataset))
    for i in list_of_indices:
        idx = torch.tensor(dataset.targets) == i
        mask = mask | idx
    mask = mask.nonzero().reshape(-1)
    return Subset(dataset, mask)

mnist_01_train = get_subset(list(range(2)), mnist_train)
mnist_01_test = get_subset(list(range(2)), mnist_test)
mnist_01_train_loader = DataLoader(mnist_01_train, batch_size=64)
mnist_01_test_loader = DataLoader(mnist_01_test, batch_size=64)
dataset_sizes = {'train': len(mnist_01_train), 'val': len(mnist_01_test)}
my_loader = {'train': mnist_01_train_loader, 'val': mnist_01_test_loader}

def create_model(init_out=2):
    model = torchvision.models.resnet18(pretrained=True)
    for param in model.parameters():
        if len(param.shape) != 1:
            param.requires_grad = False
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
    model.layer4[1].relu = nn.Tanh()
    # model.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.maxpool = nn.Identity()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, init_out)
    return model


model_ft = create_model(init_out=2).to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
# optimizer_ft = torch.optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
optimizer_ft = torch.optim.Adam(model_ft.parameters(), lr=1e-4)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=50, gamma=0.1)



def train_model(model, criterion, optimizer, scheduler, dataloaders, num_epochs=10):
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


# model_fted = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, my_loader, num_epochs=15)
# model_fted = model_fted.eval()
# model_fted.fc = nn.Identity()

model_fted = create_model(2)
model_fted.fc = nn.Identity()
model_fted.load_state_dict(torch.load('mnist_examplar_feature_extractor_full.pt'))
model_fted = model_fted.eval()
model_fted = model_fted.to(device)


mnist_train = torchvision.datasets.MNIST(root='./MNIST', train=True, transform=preprocess_test)
mnist_test = torchvision.datasets.MNIST(root='./MNIST', train=False, transform=preprocess_test)
mnist_loader_train = DataLoader(mnist_train, batch_size=64)
mnist_loader_test = DataLoader(mnist_test, batch_size=64)

with torch.no_grad():
    transformed_x_train = []
    transformed_y_train = torch.tensor([])
    for x, y in mnist_loader_train:
        transformed_x_train += [model_fted(x.to(device))]
        transformed_y_train = torch.cat((transformed_y_train, y))
    transformed_x_train = torch.vstack(transformed_x_train).cpu()
    transformed_y_train = transformed_y_train.to(torch.long)

    transformed_x_test = []
    transformed_y_test = torch.tensor([])
    for x, y in mnist_loader_test:
        transformed_x_test += [model_fted(x.to(device))]
        transformed_y_test = torch.cat((transformed_y_test, y))
    transformed_x_test = torch.vstack(transformed_x_test).cpu()
    transformed_y_test = transformed_y_test.to(torch.long)

# pd.DataFrame(transformed_x_train.numpy()).to_csv('mnist_pretrained_train.csv', index=False)
# pd.DataFrame(transformed_x_test.numpy()).to_csv('mnist_pretrained_test.csv', index=False)


import umap
a = umap.UMAP(random_state=42)
a.fit(transformed_x_test.numpy())
plt.scatter(a.embedding_[:, 0], a.embedding_[:, 1], c=transformed_y_test.numpy(), alpha=.1)
plt.savefig('umap_finetuned.png')
plt.close()




transformed_x_train = torch.tensor(pd.read_csv('mnist_pretrained_train.csv').to_numpy(), dtype=torch.float)
transformed_x_test = torch.tensor(pd.read_csv('mnist_pretrained_test.csv').to_numpy(), dtype=torch.float)
mnist_train = torchvision.datasets.MNIST(root='./MNIST', train=True)
mnist_test = torchvision.datasets.MNIST(root='./MNIST', train=False)
transformed_y_test = mnist_test.targets
transformed_y_train = mnist_train.targets

# task_split = [[0, 1, 2, 3], [4, 5], [6, 7], [8, 9]]
task_split = [[0,1], [2,3], [4,5], [6,7], [8,9]]
CIL_mnist_train = {}
CIL_mnist_test = {}
# nhwc to nchw
for _ in range(len(task_split)):
    task_id = np.array([i for i,j in enumerate(transformed_y_train) if j in task_split[_]])
    CIL_mnist_train['X_{}'.format(_)] = transformed_x_train[task_id]
    CIL_mnist_train['y_{}'.format(_)] = transformed_y_train[task_id]
    task_id = np.array([i for i, j in enumerate(transformed_y_test) if j in task_split[_]])
    CIL_mnist_test['X_{}'.format(_)] = transformed_x_test[task_id]
    CIL_mnist_test['y_{}'.format(_)] = transformed_y_test[task_id]


x_dim = CIL_mnist_train['X_0'].shape[1]


test_model = CLBruno(x_dim=x_dim, y_dim=128, task_dim=128, cond_dim=256, conv=False, task_num=1,
                     y_cat_num=[2], single_task=True, n_dense_block=4, n_hidden_dense=128,
                     activation=nn.Tanh(), mu_init=0., var_init=1., corr_init=0.1)

test_model = test_model.to('cuda')

train_loss, test_loss = test_model.train_init(CIL_mnist_train['X_0'], CIL_mnist_train['y_0'],
                                              torch.zeros(CIL_mnist_train['y_0'].shape[0], dtype=torch.long),
                                              batch_size=128, epoch=20, weight_decay=0., lr=1e-3, embedding_reg=0.1,
                                              context_portion=0.2)


N = CIL_mnist_test['X_0'].shape[0]
my_id_test = np.random.choice(range(CIL_mnist_test['X_0'].shape[0]), N, replace=False)
# runnable, check outputs
q = torch.zeros((N, 2))
p = torch.zeros(N)
for i,j in enumerate(my_id_test):
    a, b = test_model.prediction(CIL_mnist_test['X_0'][j], 0)
    q[i] = a.cpu()
    p[i] = b.cpu()
print(torch.sum(q.cpu().argmax(1) != CIL_mnist_test['y_0'][my_id_test])/N)


train_loss1, test_loss1, reg_loss1 = test_model.train_new_task(new_X=CIL_mnist_train['X_1'],
                                                               new_y=CIL_mnist_train['y_1']-2,
                                                               new_task=torch.zeros(CIL_mnist_train['y_1'].shape[0], dtype=torch.long) + 1,
                                                               epoch=20, batch_size=128, weight_decay=0.,
                                                               lr=1e-3, n_pseudo=128, embedding_reg=0.1)

N = CIL_mnist_test['X_1'].shape[0]
# alighment reg, norm_reg= 0.3 for mixture model, 1 here
# runnable, check outputs
q = torch.zeros((N, 2))
p = torch.zeros(N)
for i in range(N):
    a, b = test_model.prediction(CIL_mnist_test['X_0'][i], 0)
    q[i] = a.cpu()
    p[i] = b.cpu()
print(torch.sum(q.cpu().argmax(1) != CIL_mnist_test['y_0'][:N])/N)

# runnable, check outputs
q = torch.zeros((N, 2))
p = torch.zeros(N)
for i in range(N):
    a, b = test_model.prediction(CIL_mnist_test['X_1'][i], 1)
    q[i] = a.cpu()
    p[i] = b.cpu()
print(torch.sum(q.cpu().argmax(1) != (CIL_mnist_test['y_1'][:N]-2))/N)




train_loss2, test_loss2, reg_loss2 = test_model.train_new_task(new_X=CIL_mnist_train['X_2'],
                                                               new_y=CIL_mnist_train['y_2']-4,
                                                               new_task=torch.zeros(CIL_mnist_train['y_2'].shape[0], dtype=torch.long) + 2,
                                                               epoch=20, batch_size=128, weight_decay=0.,
                                                               lr=1e-3, n_pseudo=128, embedding_reg=0.1)

N = CIL_mnist_test['X_0'].shape[0]
# alighment reg, norm_reg= 0.3 for mixture model, 1 here
# runnable, check outputs
q = torch.zeros((N, 2))
p = torch.zeros(N)
for i in range(N):
    a, b = test_model.prediction(CIL_mnist_test['X_0'][i], 0)
    q[i] = a.cpu()
    p[i] = b.cpu()
print(torch.sum(q.cpu().argmax(1) != CIL_mnist_test['y_0'][:N])/N)

N = CIL_mnist_test['X_1'].shape[0]
# runnable, check outputs
q = torch.zeros((N, 2))
p = torch.zeros(N)
for i in range(N):
    a, b = test_model.prediction(CIL_mnist_test['X_1'][i], 1)
    q[i] = a.cpu()
    p[i] = b.cpu()
print(torch.sum(q.cpu().argmax(1) != (CIL_mnist_test['y_1'][:N]-2))/N)

N = CIL_mnist_test['X_2'].shape[0]
# runnable, check outputs
q = torch.zeros((N, 2))
p = torch.zeros(N)
for i in range(N):
    a, b = test_model.prediction(CIL_mnist_test['X_2'][i], 2)
    q[i] = a.cpu()
    p[i] = b.cpu()
print(torch.sum(q.cpu().argmax(1) != (CIL_mnist_test['y_2'][:N]-4))/N)



train_loss3, test_loss3, reg_loss3 = test_model.train_new_task(new_X=CIL_mnist_train['X_3'],
                                                               new_y=CIL_mnist_train['y_3']-6,
                                                               new_task=torch.zeros(CIL_mnist_train['y_3'].shape[0], dtype=torch.long) + 3,
                                                               epoch=15, batch_size=128, weight_decay=0.,
                                                               lr=1e-3, n_pseudo=128, embedding_reg=0.1)


train_loss4, test_loss4, reg_loss4 = test_model.train_new_task(new_X=CIL_mnist_train['X_4'],
                                                               new_y=CIL_mnist_train['y_4']-8,
                                                               new_task=torch.zeros(CIL_mnist_train['y_4'].shape[0], dtype=torch.long) + 4,
                                                               epoch=15, batch_size=128, weight_decay=0.,
                                                               lr=1e-3, n_pseudo=128, embedding_reg=0.1)


acc = 0.

for j in range(5):
    N = CIL_mnist_test['X_{}'.format(j)].shape[0]
    # alighment reg, norm_reg= 0.3 for mixture model, 1 here
    # runnable, check outputs
    q = torch.zeros((N, 2))
    p = torch.zeros(N)
    for i in range(N):
        a, b = test_model.prediction(CIL_mnist_test['X_{}'.format(j)][i], j)
        q[i] = a.cpu()
        p[i] = b.cpu()
    print(torch.sum(q.cpu().argmax(1) != (CIL_mnist_test['y_{}'.format(j)][:N]-2*j)[:N]) / N)
    acc += torch.sum(q.cpu().argmax(1) != (CIL_mnist_test['y_{}'.format(j)][:N]-2*j)[:N]) / N
print(acc/5)

torch.save(test_model.state_dict(), 'mnist_examplar_feature_extractor.pt')
torch.save(test_model.state_dict(), 'mnist_examplar_testmodel.pt')
