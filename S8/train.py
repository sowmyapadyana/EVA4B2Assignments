from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm

train_acc = []
train_losses = []


def train_model(model, device, train_loader, optimizer, epoch, regularization=None):
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        y_pred = model(data)

        output = model(data)

        loss = F.nll_loss(output, target)

        if regularization == "L1":
            # l1_crit = nn.L1Loss(size_average=False)
            reg_loss = 0
            for param in model.parameters():
                reg_loss += torch.sum(param.abs())

            factor = 0.0005
            loss += factor * reg_loss

        loss.backward()
        optimizer.step()

        pred = y_pred.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        # pbar.set_description(desc= f'loss={loss.item()} batch_id={batch_idx}')
        pbar.set_description(desc= f'loss={loss.item()} batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
        train_acc.append(100*correct/processed)