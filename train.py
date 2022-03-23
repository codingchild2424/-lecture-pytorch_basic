import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from model import DL_Model
from trainer import Trainer

def define_argparser():
    
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    
    p.add_argument('--n_epochs', type=int, default=5)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--learning_rate', type=int, default=0.001)

    config = p.parse_args()

    return config


def main(config):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    train_dataloader = DataLoader(training_data, batch_size= config.batch_size)
    test_dataloader = DataLoader(test_data, batch_size= config.batch_size)

    model = DL_Model().to(device)
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr= config.learning_rate)

    trainer = Trainer(device, model, loss_fn, optimizer, config)

    trainer.trainer(train_dataloader, test_dataloader)

if __name__ == "__main__":
    config = define_argparser()
    main(config)