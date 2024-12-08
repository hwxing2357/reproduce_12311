from torch import Tensor, mul, normal, zeros, full, cat, no_grad, eq
from torch.nn import Module, Sequential, LeakyReLU, Conv2d, BatchNorm2d, Linear, Sigmoid, Softmax, Dropout2d, Embedding
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.init import normal_, constant_
# taken from https://github.com/jacons/MeRGAN-V2/tree/main
class Discriminator(Module):
    def __init__(self, channels: int, classes: int = 10):
        super(Discriminator, self).__init__()

        self.conv_blocks = Sequential(

            nn.Linear(channels, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

        # Output layers
        self.discr = Sequential(
            Linear(128, 1),
            Sigmoid())

        self.classifier = Sequential(
            Linear(128, classes),
            Softmax(dim=1))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.conv_blocks(x)
        x = x.view(x.shape[0], -1)

        return self.discr(x), self.classifier(x)


class Generator(Module):
    def __init__(self, channels: int = 1, num_classes: int = 10, embedding_dim: int = 100):
        super(Generator, self).__init__()

        self.embedding_dim = embedding_dim
        self.embedding = Embedding(num_classes, embedding_dim)
        self.fc = Linear(embedding_dim, embedding_dim)

        self.conv_blocks = Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, channels)
        )

    def forward(self, labels: Tensor, z: Tensor = None) -> Tensor:
        batch_size = labels.size(0)

        # If not provided, we generate the random noise
        if z is None:
            z = normal(0, 1, (batch_size, self.embedding_dim), device=labels.device)

        # concat with the conditional label
        input_ = mul(self.embedding(labels), z)

        out = self.fc(input_)
        return self.conv_blocks(out)


class ExperienceDataset(Dataset):

    def __init__(self, image: Tensor, target: Tensor, device: str = "cpu"):
        self.image, self.target = image, target
        self.device = device

    def __len__(self):
        return self.image.size(0)

    def __getitem__(self, idx):
        return self.image[idx].to(self.device), self.target[idx].to(self.device)


def weights_init_normal(m):
    class_name = m.__class__.__name__
    if class_name.find("Conv") != -1:
        normal_(m.weight.data, 0.0, 0.02)

    elif class_name.find("BatchNorm2d") != -1:
        normal_(m.weight.data, 1.0, 0.02)
        constant_(m.bias.data, 0.0)


def compute_acc(predicted: Tensor, labels: Tensor):
    """
    Compute the accuracy of model both for real and fake images
    :param predicted: label predicted by discriminator
    :param labels:  true label
    :return:
    """
    correct = eq(predicted.argmax(dim=1), labels).sum().item()
    return float(correct) / float(labels.size(0))


import numpy as np
import pandas as pd

class Join_retrain:
    def __init__(self, generator: Generator, batch_size: int, buff_img: int, channels: int,
                 device: str = "cpu"):
        """
        Manage the join retraining in an online setting
        """
        self.g = generator
        self.batch_size = batch_size
        self.buff_img = buff_img
        self.device = device
        self.channels = channels

    def create_buffer(self, id_exp: int, past_classes: Tensor,
                      source: tuple[Tensor, Tensor]) -> DataLoader:

        real_image, real_label = source
        device = self.device

        if id_exp == 0:  # No previous experience (first experience)
            return DataLoader(ExperienceDataset(real_image, real_label, device),
                              shuffle=True,
                              batch_size=self.batch_size)

        elif id_exp > 0:  # generating buffer replay

            # Define the number of images to generate, we allocate a fixed number of slots for each number of class
            # encountered
            img_to_create = self.buff_img * past_classes.size(0)
            gen_buffer = zeros((img_to_create, self.channels), device=self.device)

            self.g.eval()
            with no_grad():
                count = 0
                for i in past_classes:  # for each class encountered

                    # since the buffer may have high dimension, we generate image in batch fashion
                    to_generate = self.buff_img
                    while to_generate > 0:
                        batch_size = min(256, to_generate)
                        gen_label = full((batch_size,), i, device=device)
                        gen_buffer[count:count + batch_size] = self.g(gen_label)

                        count += batch_size
                        to_generate -= batch_size
            self.g.train()

            # In the end, we concat the replay generated and the current batch of image (new classes)
            custom_x = cat((real_image, gen_buffer.cpu()), dim=0)
            custom_y = cat(
                (real_label, past_classes.repeat_interleave(self.buff_img)),
                dim=0)

            return DataLoader(ExperienceDataset(custom_x, custom_y, device),
                              shuffle=True,
                              batch_size=self.batch_size)

#
import copy
import os
from typing import Dict
import matplotlib.pyplot as plt
from torch import arange, ones, zeros, randint, cat, tensor, stack, normal, Tensor, no_grad
from torch.nn import BCELoss, CrossEntropyLoss, MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader
# from torchvision.utils import make_grid, save_image
from tqdm import tqdm


class Trainer:
    def __init__(self, config: Dict, generator: Generator = None, discriminator: Discriminator = None):

        # Retrieve the parameters
        self.device, self.n_epochs = config["device"], config["n_epochs"]
        self.img_size, self.embedding_dim = config["img_size"], config["embedding"]
        self.channels, self.batch_size = config["channels"], config["batch_size"]
        self.num_classes = config["num_classes"]

        # Set variables for continual evaluation: fixed noise and labels
        self.eval_noise = normal(0, 1, (100, self.embedding_dim), device=self.device)
        self.eval_label = arange(0, self.num_classes).repeat(10).to(self.device)

        # Define the generator and discriminator if they are not provided
        if generator is None or discriminator is None:

            self.generator = Generator(
                num_classes=config["num_classes"],
                embedding_dim=config["embedding"],
                channels=self.channels
            ).to(self.device)

            self.discriminator = Discriminator(
                classes=config["num_classes"],
                channels=self.channels,
            ).to(self.device)
            # Initialize the weights
            self.discriminator.apply(weights_init_normal)
            self.generator.apply(weights_init_normal)
        else:
            self.generator = generator.to(self.device)
            self.discriminator = discriminator.to(self.device)

        # Loss functions and optimizers
        self.adversarial_loss = BCELoss().to(self.device)
        self.auxiliary_loss = CrossEntropyLoss().to(self.device)
        self.optimizer_g = Adam(self.generator.parameters(), lr=config["lr_g"], betas=(0.5, 0.999))
        self.optimizer_d = Adam(self.discriminator.parameters(), lr=config["lr_d"], betas=(0.5, 0.999))

    def fit_classic(self, experiences, create_gif: bool = False, const_gen: float = 0.5, const_dis: float = 0.25,
                    folder: str = "classical_acgan") -> Tensor:

        if create_gif:
            os.makedirs(folder, exist_ok=True)

        device, n_epochs, batch_size_ = self.device, self.n_epochs, self.batch_size
        loss_history = []

        for idx, (classes, x, y) in enumerate(experiences):  # for each experience
            # Oss. "Classes" are a list of targets in the batch

            current_classes = tensor(classes)  # Number that can be generated
            print("-- Experience -- ", idx + 1, "classes", current_classes.tolist())

            loader = DataLoader(ExperienceDataset(x, y, device), shuffle=True, batch_size=batch_size_)
            for epoch in range(0, n_epochs[idx]):
                for batch, (real_image, real_label) in enumerate(tqdm(loader)):
                    batch_size = real_image.size(0)

                    valid = ones((batch_size, 1), device=device)
                    fake = zeros((batch_size, 1), device=device)

                    # ---- Generator ----
                    self.optimizer_g.zero_grad()

                    gen_label = current_classes[randint(0, len(current_classes), size=(batch_size,))].to(device)
                    fake_img = self.generator(gen_label)

                    dis_output, aux_output = self.discriminator(fake_img)
                    errG = const_gen * (
                            self.adversarial_loss(dis_output, valid) +
                            self.auxiliary_loss(aux_output, gen_label))

                    errG.backward()
                    self.optimizer_g.step()
                    # ---- Generator ----

                    # ---- Discriminator ----
                    self.optimizer_d.zero_grad()

                    dis_real, aux_real = self.discriminator(real_image)
                    dis_fake, aux_fake = self.discriminator(fake_img.detach())

                    errD = const_dis * (
                            self.adversarial_loss(dis_real, valid) +
                            self.adversarial_loss(dis_fake, fake) +
                            self.auxiliary_loss(aux_real, real_label) +
                            self.auxiliary_loss(aux_fake, gen_label)
                    )

                    errD.backward()
                    self.optimizer_d.step()
                    # ---- Discriminator ----

                    d_acc = compute_acc(
                        cat([aux_real, aux_fake], dim=0),
                        cat([real_label, gen_label], dim=0)
                    )

                    loss_history.append(tensor([errD.item(), errG.item(), d_acc]))

                print("[%d/%d] Loss_D: %.4f Loss_G: %.4f Acc %.6f"
                      % (epoch + 1, n_epochs[idx], loss_history[-1][0], loss_history[-1][1],
                         loss_history[-1][2]))

        return stack(loss_history).T

    def fit_join_retrain(self, experiences, buff_img: int, create_gif: bool = False, const_gen: float = 0.5,
                         const_dis: float = 0.25, folder: str = "join_retrain") -> Tensor:

        if create_gif:
            os.makedirs(folder, exist_ok=True)

        device, n_epochs = self.device, self.n_epochs
        loss_history = []
        current_classes = None  # Tensor of current classes

        jr = Join_retrain(generator=self.generator,
                          batch_size=self.batch_size,
                          buff_img=buff_img,
                          channels=self.channels,
                          device=device)

        for idx, (classes, x, y) in enumerate(experiences):
            # Oss. "Classes" are a list of targets in the batch

            """
            0ss2.
            In the first experience we passed "create_buffer" a "current_class" that it is equal to None, 
            but it is ok, because the first experience there is not a buffer replay. The second and
            further experiences, the "current_class" (at this line) is not updated and so it refers to
            the previous classes.            
            """
            loader = jr.create_buffer(idx, current_classes, (x, y))

            new_classes = tensor(classes)  # Transform into tensor the classes list

            # In this case, we concatenate the past classes with the current ones
            current_classes = new_classes if current_classes is None else cat((current_classes, new_classes))
            print("-- Experience -- ", idx + 1, "numbers", current_classes.tolist())

            for epoch in range(0, n_epochs[idx]):
                for real_image, real_label in tqdm(loader):
                    batch_size = real_image.size(0)

                    valid, fake = ones((batch_size, 1), device=device), zeros((batch_size, 1), device=device)

                    # ---- Generator ----
                    self.optimizer_g.zero_grad()

                    gen_label = current_classes[randint(0, len(current_classes), size=(batch_size,))].to(device)
                    fake_img = self.generator(gen_label)

                    dis_output, aux_output = self.discriminator(fake_img)
                    errG = const_gen * (
                            self.adversarial_loss(dis_output, valid) +
                            self.auxiliary_loss(aux_output, gen_label))

                    errG.backward()
                    self.optimizer_g.step()
                    # ---- Generator ----

                    # ---- Discriminator ----
                    self.optimizer_d.zero_grad()

                    dis_real, aux_real = self.discriminator(real_image)
                    dis_fake, aux_fake = self.discriminator(fake_img.detach())

                    errD = const_dis * (
                            self.adversarial_loss(dis_real, valid) +
                            self.adversarial_loss(dis_fake, fake) +
                            self.auxiliary_loss(aux_real, real_label) +
                            self.auxiliary_loss(aux_fake, gen_label)
                    )

                    errD.backward()
                    self.optimizer_d.step()
                    # ---- Discriminator ----

                    d_acc = compute_acc(
                        cat([aux_real, aux_fake], dim=0),
                        cat([real_label, gen_label], dim=0)
                    )

                    loss_history.append(tensor([errD.item(), errG.item(), d_acc]))

                print("[%d/%d] Loss_D: %.4f Loss_G: %.4f Acc %.6f"
                      % (epoch + 1, n_epochs[idx], loss_history[-1][0], loss_history[-1][1],
                         loss_history[-1][2]))

        return stack(loss_history).T

    def fit_replay_alignment(self, experiences, create_gif: bool = False, const_gen: float = 0.5,
                             const_dis: float = 0.25, const_ra: float = 1, folder: str = "replay_alignment"):
        if create_gif:
            os.makedirs(folder, exist_ok=True)

        device, n_epochs, batch_size_ = self.device, self.n_epochs, self.batch_size
        history = []

        current_classes = None  # Tensor of current classes (new classes)
        prev_classes = None  # Tensor of previous classes (concatenated)
        prev_gen = None  # Generator in the previous experience

        alignment_loss = MSELoss().to(self.device)

        for idx, (classes, x, y) in enumerate(experiences):

            """
            Oss. The mechanism is similar to the previous one, but.. 
            For the first epoch, we train as a classical acGAN. Then we train the model only with the current classes,
            but the "alignment" is performed with the past classes".
            """
            if idx > 0:
                prev_classes = copy.deepcopy(current_classes) if prev_classes is None else cat(
                    (prev_classes, current_classes))

            current_classes = tensor(classes)  # Transform into tensor the classes list

            print("-- Experience -- ", idx + 1, "numbers", current_classes.tolist())
            if prev_classes is not None:
                print("Past experiences", prev_classes.tolist())

            loader = DataLoader(ExperienceDataset(x, y, device), shuffle=True, batch_size=batch_size_)
            for epoch in range(0, n_epochs[idx]):
                for real_image, real_label in tqdm(loader):
                    batch_size = real_image.size(0)

                    valid, fake = ones((batch_size, 1), device=device), zeros((batch_size, 1), device=device)

                    # ---- Generator ----
                    self.optimizer_g.zero_grad()

                    gen_label = current_classes[randint(0, len(current_classes), size=(batch_size,))].to(device)
                    fake_img = self.generator(gen_label)

                    dis_output, aux_output = self.discriminator(fake_img)

                    # ---------------- replay alignment ----------------
                    align_loss = 0
                    if prev_gen is not None:
                        z = normal(0, 1, (batch_size_, self.embedding_dim), device=device)
                        gen_label_ = prev_classes[randint(0, len(prev_classes), size=(batch_size_,))].to(device)

                        fake_img1 = self.generator(gen_label_, z)
                        with no_grad():
                            fake_img2 = prev_gen(gen_label_, z)

                        align_loss = alignment_loss(fake_img1, fake_img2)
                    # ---------------- replay alignment ----------------

                    errG = const_gen * (
                            self.adversarial_loss(dis_output, valid) +
                            self.auxiliary_loss(aux_output, gen_label)
                    ) + const_ra * align_loss

                    errG.backward()
                    self.optimizer_g.step()
                    # ---- Generator ----

                    # ---- Discriminator ----
                    self.optimizer_d.zero_grad()

                    dis_real, aux_real = self.discriminator(real_image)
                    dis_fake, aux_fake = self.discriminator(fake_img.detach())

                    errD = const_dis * (
                            self.adversarial_loss(dis_real, valid) +
                            self.adversarial_loss(dis_fake, fake) +
                            self.auxiliary_loss(aux_real, real_label) +
                            self.auxiliary_loss(aux_fake, gen_label)
                    )
                    errD.backward()
                    self.optimizer_d.step()
                    # ---- Discriminator ----

                    d_acc = compute_acc(
                        cat([aux_real, aux_fake], dim=0),
                        cat([real_label, gen_label], dim=0)
                    )
                    history.append(tensor([errD.item(), errG.item(), d_acc]))

                print("[%d/%d] Loss_D: %.4f Loss_G: %.4f Acc %.6f"
                      % (epoch + 1, n_epochs[idx], history[-1][0], history[-1][1],
                         history[-1][2]))

            prev_gen = copy.deepcopy(self.generator)
            prev_gen.eval()

        return stack(history).T

def plot_history(history: Tensor):
    fig = plt.figure(figsize=(25, 14))
    sub_figs = fig.subfigures(1, 2)

    ax_img = sub_figs[0].subplots(nrows=2)
    ax_img[0].plot(history[0], label="Discriminator loss", color="blue", alpha=0.8)
    ax_img[0].grid()
    ax_img[0].legend()
    ax_img[0].set_xlabel("Updates")
    ax_img[0].set_ylabel("Loss value")
    ax_img[0].set_title("Discriminator loss")

    ax_img[1].plot(history[1], label="Generator loss", color="green", alpha=0.8)
    ax_img[1].grid()
    ax_img[1].legend()
    ax_img[1].set_xlabel("Updates")
    ax_img[1].set_ylabel("Loss value")
    ax_img[1].set_title("Generator loss")

    ax_c = sub_figs[1].subplots()
    ax_c.plot(history[2], label="Accuracy", color="orange", alpha=0.8)
    ax_c.grid()
    ax_c.legend()
    ax_c.set_xlabel("Updates")
    ax_c.set_ylabel("Accuracy")
    ax_c.set_title("Accuracy")

    plt.tight_layout(pad=6)
    plt.show()


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

my_data = [[label_grouping[i], PAN_CAN_data_train[i], PAN_CAN_label_train[i]] for i in range(len(label_grouping))]


config = dict(
    device="cuda" if torch.cuda.is_available() else "cpu",
    num_classes=32,
    img_size=None,
    channels=2000,
    n_epochs=[200,200, 300, 300, 400, 400],
    batch_size=64,
    embedding=100, # latent dimension of embedding
    lr_g=1e-4, # Learning rate for generator 7e-5
    lr_d=1e-4 # Learning rate for discriminator
)



trainer = Trainer(config=config)
history = trainer.fit_replay_alignment(experiences= my_data, const_ra=0.8)

plot_history(history)
