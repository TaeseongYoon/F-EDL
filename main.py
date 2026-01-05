import time
import argparse
import json
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch

from train import train, eval, conf_calibration, ood_detection
from datasets import load_datasets
from models import FEDL


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ID_dataset",
        default="CIFAR-10",
        choices=["MNIST", "CIFAR-10", "CIFAR-100"],
        help="Pick a dataset",
    )

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--val_size", type=float, default=0.05)

    parser.add_argument("--imbalance_factor", type=float, default=0)
    parser.add_argument("--noise", action="store_true")

    parser.add_argument("--dropout_rate", type=float, default=0)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--step_size", type=int, default=20)

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--result_dir", type=str, default="saved_results")
    parser.add_argument("--model_dir", type=str, default="saved_models")

    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)

    return parser.parse_args()


def to_python(x):
    if isinstance(x, torch.Tensor):
        return x.item()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, (list, tuple)):
        return [to_python(elem) for elem in x]
    if isinstance(x, dict):
        return {k: to_python(v) for k, v in x.items()}
    try:
        return float(x)
    except Exception:
        return x


def main(args):
    print(args)

    trainloader, validloader, testloader, ood_loader1, ood_loader2 = load_datasets(
        args.ID_dataset,
        args.batch_size,
        args.val_size,
        args.imbalance_factor,
        args.noise,
    )

    model = FEDL(
        args.ID_dataset,
        args.dropout_rate,
        args.device,
        args.hidden_dim,
        args.num_layers,
    )

    train(
        model,
        args.learning_rate,
        args.weight_decay,
        args.step_size,
        args.num_epochs,
        trainloader,
        validloader,
        args.device,
    )

    top1_acc, top2_acc = eval(model, testloader, args.device)
    conf_auroc, conf_aupr, brier = conf_calibration(model, testloader, args.device)
    ood_auroc, ood_aupr = ood_detection(
        model, testloader, ood_loader1, ood_loader2, args.device
    )

    result = {
        "Top-1 Accuracy": to_python(top1_acc),
        "Top-2 Accuracy": to_python(top2_acc),
        "CONF AUROC": to_python(conf_auroc),
        "CONF AUPR": to_python(conf_aupr),
        "BRIER": to_python(brier),
        "OOD AUROC": to_python(ood_auroc),
        "OOD AUPR": to_python(ood_aupr),
    }

    timestamp = int(time.time())

    result_filename = (
        f"{args.ID_dataset}_IR{args.imbalance_factor}_LR{args.learning_rate}"
        f"_HD{args.hidden_dim}_NL{args.num_layers}_results_{timestamp}.json"
    )

    model_filename = (
        f"{args.ID_dataset}_IR{args.imbalance_factor}_LR{args.learning_rate}"
        f"_HD{args.hidden_dim}_NL{args.num_layers}_model_{timestamp}.pth"
    )

    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    result_path = os.path.join(args.result_dir, result_filename)
    with open(result_path, "w") as f:
        json.dump(result, f, indent=4)

    model_path = os.path.join(args.model_dir, model_filename)
    torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)
