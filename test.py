import torch
from torchvision.models import mobilenet_v2
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.nn as nn

import os
from tqdm import tqdm
from glob import glob
from sklearn.metrics import classification_report


DEVICE = 6
ROOT_DIR = "/raid/n.kotov1/Dataset_bpla_patches"
RESULTS_DIR = "results"
NET_NAME = "mobilenet_v2"
WEIGHT_DIR = "weight"

if not os.path.exists(os.path.join(RESULTS_DIR, NET_NAME)): os.makedirs(os.path.join(RESULTS_DIR, NET_NAME))


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '3,4'
    device = torch.device("cuda")

    # device = torch.cuda.set_device(DEVICE)
    # device = torch.device(DEVICE)
    
    test_transform = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Resize(224),
                           transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                       ])

    test_data = datasets.ImageFolder(root=os.path.join(ROOT_DIR, 'test'), 
                                  transform=test_transform)
    
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=1,
        shuffle=False,
        num_workers=4
    )

    all_weights = sorted(glob(os.path.join(WEIGHT_DIR, NET_NAME, '*.pth'), recursive=True))

    for w in tqdm(all_weights, total=len(all_weights)):
        model = mobilenet_v2(pretrained=True)
        model._modules['classifier'][-1] = nn.Linear(model._modules['classifier'][-1].in_features, len(test_data.classes), bias=True)
        model.load_state_dict(torch.load(w))
        model = nn.DataParallel(model)
        model = model.to(device)
        
        images, labels, probs = test(model, device, test_loader)
        pred_labels = torch.argmax(probs, 1)

        with open(os.path.join(RESULTS_DIR, NET_NAME, f"{w.split('/')[-1].split('.')[0]}.txt"), "a") as f:
            f.write(f"\n {w} \n")
            f.writelines(classification_report(labels, pred_labels, target_names=test_data.classes))


def test(model, device, loader):
    model.eval()
    
    images = []
    labels = []
    probs = []

    with torch.no_grad():
        for (x, y) in tqdm(loader, total=len(loader)):
            x = x.to(device)
            y_pred = model(x)
            y_prob = F.softmax(y_pred, dim=-1)
            top_pred = y_prob.argmax(1, keepdim=True)
            images.append(x.cpu())
            labels.append(y.cpu())
            probs.append(y_prob.cpu())
    
    images = torch.cat(images, dim=0)
    labels = torch.cat(labels, dim=0)
    probs = torch.cat(probs, dim=0)
    
    return images, labels, probs


if __name__ == "__main__":
    main()
