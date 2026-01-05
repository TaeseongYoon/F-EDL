import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import uq 
from uq import *

try:
    from torch.cuda.amp import GradScaler, autocast
    _supports_device_type = "device_type" in autocast.__init__.__code__.co_varnames
except Exception:
    from torch.cuda.amp import GradScaler, autocast
    _supports_device_type = False


def train(model, learning_rate, weight_decay, step_size, num_epochs, trainloader, validloader, device):
    """Training loop with mixed precision and simple early stopping."""
    scaler = GradScaler()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)

    VAL_ACC, VAL_LOSS = [], []
    patience, best_acc, epochs_no_improve = 3, 0.0, 0

    model = model.to(device)
    model.train()

    for epoch in tqdm(range(num_epochs)):
        running_loss = 0.0

        for x, y in trainloader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            num_classes = 10
            y_oh = F.one_hot(y, num_classes).float()

            optimizer.zero_grad(set_to_none=True)

            # ? Mixed precision training (handle both old/new torch versions)
            if _supports_device_type:
                context = autocast(device_type='cuda')
            else:
                context = autocast()

            with context:
                alpha, p, tau = model(x)
                mu, var = compute_moments(alpha, p, tau)

                # Main classification loss
                loss_cls = torch.sum((y_oh - mu) ** 2) + torch.sum(var)
                # Auxiliary regularization for p
                loss_reg_p = torch.sum((y_oh - p) ** 2)

                loss = loss_cls + loss_reg_p

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        scheduler.step()

        if epoch % 10 == 0 and epoch > 0:
            model.eval()
            val_loss, correct_v, total_v = 0.0, 0, 0

            with torch.no_grad():
                for x_v, y_v in validloader:
                    x_v, y_v = x_v.to(device), y_v.to(device)
                    y_oh_v = F.one_hot(y_v, num_classes).float()

                    alpha_v, p_v, tau_v = model(x_v)
                    mu_v, var_v = compute_moments(alpha_v, p_v, tau_v)

                    loss_cls_v = torch.sum((y_oh_v - mu_v) ** 2) + torch.sum(var_v)
                    loss_reg_p_v = torch.sum((y_oh_v - p_v) ** 2)

                    val_loss += (loss_cls_v + loss_reg_p_v).item()

                    y_pred_v = mu_v.argmax(1)
                    correct_v += (y_pred_v == y_v).sum().item()
                    total_v += y_v.size(0)

            val_acc = 100 * correct_v / total_v
            VAL_ACC.append(val_acc)
            VAL_LOSS.append(val_loss)

            print(f"Epoch {epoch:3d} | Val Loss = {val_loss:.3f}, Val Acc = {val_acc:.2f}%")

            # Early stopping logic
            if val_acc > best_acc + 0.01:
                best_acc = val_acc
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

            model.train()


def eval(model, testloader, device):
    """Evaluate Top-1 and Top-2 accuracy on the test set."""
    model.eval()
    total_t = correct_top1 = correct_top2 = 0

    with torch.no_grad():
        for x_t, y_t in testloader:
            x_t, y_t = x_t.to(device), y_t.to(device)

            alpha_t, p_t, tau_t = model(x_t)
            mu_t, _ = compute_moments(alpha_t, p_t, tau_t)

            # Top-1 accuracy
            y_pred_top1 = mu_t.argmax(dim=1)
            correct_top1 += (y_pred_top1 == y_t).sum().item()

            # Top-2 accuracy
            top2_preds = torch.topk(mu_t, k=2, dim=1).indices
            correct_top2 += sum(y_t[i].item() in top2_preds[i] for i in range(len(y_t)))

            total_t += y_t.size(0)

    top1_acc = 100 * correct_top1 / total_t
    top2_acc = 100 * correct_top2 / total_t

    print(f"Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"Top-2 Accuracy: {top2_acc:.2f}%")

    return top1_acc, top2_acc
