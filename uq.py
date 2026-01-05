import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn import metrics
from torch.utils.data import TensorDataset
from torchvision import transforms
from PIL import Image

import numpy as np
import torch
import sklearn

###############################################
# [1] Uncertainty Measures
###############################################
def check(x):
    if isinstance(x, np.ndarray):
        x_tensor = torch.tensor(x)
    else:
        x_tensor = x
    nan = torch.sum(torch.isnan(x_tensor))
    inf = torch.sum(torch.isinf(x_tensor))
    if (inf + nan) != 0:
        x_tensor = torch.nan_to_num(x_tensor)
    return x_tensor

def EU(mu, var):
    return np.sum(var, axis=1)

def TU(mu, var):
    return 1 - np.sum(mu ** 2, axis=1)

def AU(mu, var):
    return TU(mu, var) - EU(mu, var)


########################
# F-EDL Moments
########################
def compute_moments(alpha, p, tau):
    alpha0 = alpha.sum(dim=1, keepdim=True)
    mu = (alpha + tau * p) / (alpha0 + tau)
    var = mu * (1 - mu) / (alpha0 + tau + 1) + (tau**2) * p * (1 - p) / ((alpha0 + tau) * (alpha0 + tau + 1))
    return mu, var

def get_mean_var(model, loader, device):
    MU, VAR, ALPHA, P, TAU = [], [], [], [], []
    with torch.no_grad():
        for i, (x_t, y_t) in enumerate(loader):
            alpha_t, p_t, tau_t = model(x_t.to(device))
            mu_t, var_t = compute_moments(alpha_t, p_t, tau_t)
            
            MU.append(mu_t)
            VAR.append(var_t)
            P.append(p_t)
            ALPHA.append(alpha_t)
            TAU.append(tau_t)
            
    MU, VAR = torch.cat(MU).cpu().numpy(), torch.cat(VAR).cpu().numpy()
    ALPHA, P, TAU = torch.cat(ALPHA).cpu().numpy(), torch.cat(P).cpu().numpy(), torch.cat(TAU).cpu().numpy()
    return MU, VAR, ALPHA, P, TAU


def auroc_aupr(unc_id, unc_ood):
    unc_id = check(unc_id)
    unc_ood = check(unc_ood)
    bin_labels = np.concatenate([np.ones(len(unc_id)), np.zeros(len(unc_ood))])
    scores = -np.concatenate((unc_id, unc_ood))
    auroc = metrics.roc_auc_score(bin_labels, scores)
    aupr = metrics.average_precision_score(bin_labels, scores)
    return auroc, aupr


########################
# Confidence Calibration
########################
def conf_calibration(model, testloader, device):
    CORRECT, brier_scores = [], []

    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            alpha, p, tau = model(x)
            mu, _ = compute_moments(alpha, p, tau)
            y_pred = mu.argmax(1).cpu().numpy()
            correct = (y_pred == y.cpu().numpy()).astype(int)
            y_oh = F.one_hot(y, num_classes=p.shape[1]).to(device)
            brier_batch = torch.mean((y_oh - mu) ** 2, dim=1)
            brier_scores.extend(brier_batch.cpu().numpy())
            CORRECT.append(correct)

    BRIER = np.mean(brier_scores)
    CORRECT = np.concatenate(CORRECT)
    MU_id, VAR_id, _, _, _ = get_mean_var(model, testloader, device)
    alea_id, epis_id = -AU(MU_id, VAR_id), -EU(MU_id, VAR_id)

    AUROC = {"AU": metrics.roc_auc_score(CORRECT, alea_id),
             "EU": metrics.roc_auc_score(CORRECT, epis_id)}
    AUPR = {"AU": metrics.average_precision_score(CORRECT, alea_id),
            "EU": metrics.average_precision_score(CORRECT, epis_id)}

    return AUROC, AUPR, BRIER


########################
# OOD Detection
########################
def ood_detection(model, testloader, ood_loader1, ood_loader2, device):
    MU_id, VAR_id, _, _, _ = get_mean_var(model, testloader, device)
    MU_ood1, VAR_ood1, _, _, _ = get_mean_var(model, ood_loader1, device)
    MU_ood2, VAR_ood2, _, _, _ = get_mean_var(model, ood_loader2, device)

    alea_id, epis_id = AU(MU_id, VAR_id), EU(MU_id, VAR_id)
    alea_ood1, epis_ood1 = AU(MU_ood1, VAR_ood1), EU(MU_ood1, VAR_ood1)
    alea_ood2, epis_ood2 = AU(MU_ood2, VAR_ood2), EU(MU_ood2, VAR_ood2)

    AUROC = [
        {"AU": auroc_aupr(alea_id, alea_ood1)[0], "EU": auroc_aupr(epis_id, epis_ood1)[0]},
        {"AU": auroc_aupr(alea_id, alea_ood2)[0], "EU": auroc_aupr(epis_id, epis_ood2)[0]}
    ]
    AUPR = [
        {"AU": auroc_aupr(alea_id, alea_ood1)[1], "EU": auroc_aupr(epis_id, epis_ood1)[1]},
        {"AU": auroc_aupr(alea_id, alea_ood2)[1], "EU": auroc_aupr(epis_id, epis_ood2)[1]}
    ]
    return AUROC, AUPR


########################
# Distribution Shift Detection (MNIST)
########################
def dist_shift_detection_fedl_mnist(model, testloader, device):
    normalize = transforms.Normalize((0.5,), (0.5,))
    corruptions = ['shot_noise','impulse_noise','glass_blur','motion_blur','shear','scale','rotate','brightness','translate',
                   'stripe','fog','spatter','dotted_line','zigzag','canny_edges']

    AUROC1, AUPR1, AUROC2, AUPR2 = [], [], [], []
    MU_id, VAR_id, _, _, _ = get_mean_var(model, testloader, device)
    alea_id, epis_id = AU(MU_id, VAR_id), EU(MU_id, VAR_id)

    for ctype in corruptions:
        base_path = "data/mnist_c/"
        data = torch.tensor(np.load(f"{base_path}/{ctype}/test_images.npy")).reshape(-1,1,28,28)
        data = normalize(data / 255.0)
        labels = torch.tensor(np.load(f"{base_path}/{ctype}/test_labels.npy"))
        corrupted_loader = torch.utils.data.DataLoader(TensorDataset(data, labels), batch_size=64, shuffle=False)
        MU_ood, VAR_ood, _, _, _ = get_mean_var(model, corrupted_loader, device)
        alea_ood, epis_ood = AU(MU_ood, VAR_ood), EU(MU_ood, VAR_ood)

        for unc_id, unc_ood, auc_list, aupr_list in [(alea_id, alea_ood, (AUROC1, AUPR1)), (epis_id, epis_ood, (AUROC2, AUPR2))]:
            auroc, aupr = auroc_aupr(unc_id, unc_ood)
            auc_list[0].append(auroc)
            aupr_list[1].append(aupr)

    AUROC = {"AU": np.mean(AUROC1), "EU": np.mean(AUROC2)}
    AUPR = {"AU": np.mean(AUPR1), "EU": np.mean(AUPR2)}
    return AUROC, AUPR


########################
# Distribution Shift Detection (CIFAR)
########################
class CIFARCorruption(torch.utils.data.Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.transform(Image.fromarray(self.data[idx])), 0


def dist_shift_detection_cifar(ID_dataset, model, testloader, fix_tau, fix_p, device):
    corruptions =["gaussian_noise","shot_noise","impulse_noise","defocus_blur","glass_blur","motion_blur","zoom_blur",
"snow","frost","fog","brightness","contrast","elastic_transform","pixelate","jpeg_compression",
                   "speckle_noise","gaussian_blur","spatter","saturate"]

    base_path = "data/CIFAR-10-C" if ID_dataset == "CIFAR-10" else "data/CIFAR-100-C"
    
    mean_std = ([0.4914,0.4822,0.4465],[0.2023,0.1994,0.2010]) if ID_dataset=="CIFAR-10" else ([0.5071,0.4867,0.4408],[0.2675,0.2565,0.2761])
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*mean_std)])

    MU_id, VAR_id, _, _, _ = get_mean_var(model, testloader, device)
    alea_id, total_id = AU(MU_id, VAR_id), TU(MU_id, VAR_id)

    AUROC, AUPR = {}, {}
    for corruption in corruptions:
        
        path = os.path.join(base_path, f"{corruption}.npy")
        if not os.path.exists(path): continue
            
        data = np.load(path)
        dataset = CIFARCorruption(data, transform)
        AUROC[corruption], AUPR[corruption] = {}, {}
        
        for severity in range(1, 6):
            subset = torch.utils.data.Subset(dataset, list(range(10000*(severity-1),10000*severity)))
            loader = torch.utils.data.DataLoader(subset, batch_size=128, shuffle=False)
            
            MU_ood, VAR_ood, _, _, _ = get_mean_var(model, loader, device)
            alea_ood, total_ood = AU(MU_ood, VAR_ood), TU(MU_ood, VAR_ood)
            
            AUROC[corruption][severity] = {"AU": auroc_aupr(alea_id, alea_ood)[0], "TU": auroc_aupr(total_id, total_ood)[0]}
            AUPR[corruption][severity] = {"AU": auroc_aupr(alea_id, alea_ood)[1], "TU": auroc_aupr(total_id, total_ood)[1]}
    return AUROC, AUPR



def dist_shift_detection(ID_dataset, model, testloader, fix_tau, fix_p, device):
    if ID_dataset == "MNIST":
        return dist_shift_detection_fedl_mnist(model, testloader, fix_tau, fix_p, device)
    else:
        return dist_shift_detection_cifar(ID_dataset, model, testloader, fix_tau, fix_p, device)
