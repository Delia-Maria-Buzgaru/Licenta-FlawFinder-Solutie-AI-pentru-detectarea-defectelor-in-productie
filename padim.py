import torch
import torch.nn as nn
from torchvision import models
import numpy as np
import pickle
from scipy.spatial.distance import mahalanobis

class PaDiMDetector:
    def __init__(self, device="cpu"):
        self.device = device
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to(device)
        self.model.eval()
        
        self.mean = None
        self.inv_covariance = None
        self.idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] 

    def extrage_feature_map(self, x):
        outputs = []
        def hook(module, input, output):
            outputs.append(output)
        
        h1 = self.model.layer1.register_forward_hook(hook)
        h2 = self.model.layer2.register_forward_hook(hook)
        h3 = self.model.layer3.register_forward_hook(hook)
        
        with torch.no_grad():
            self.model(x)
        
        h1.remove(); h2.remove(); h3.remove()
        
        for i in range(3):
            outputs[i] = torch.nn.functional.interpolate(outputs[i], size=(32, 32), mode='bilinear', align_corners=False)
        
        return torch.cat(outputs, 1)

    def train(self, dataloader):
        print("Antrenare PaDiM (calculare distribuție Gaussiană)...")
        features_list = []
        
        for batch in dataloader:
            batch = batch.to(self.device)
            f_map = self.extrage_feature_map(batch)
            features_list.append(f_map.cpu())
        
        features = torch.cat(features_list, 0).numpy() 
        n, c, h, w = features.shape
        
        
        features = features.reshape(n, c, h * w)
        self.mean = np.mean(features, axis=0) 
        
        self.inv_covariance = np.zeros((h * w, c, c))
        identity = np.identity(c)
        for i in range(h * w):
            self.inv_covariance[i] = np.linalg.inv(np.cov(features[:, :, i], rowvar=False) + 0.01 * identity)
            
    def predict(self, x):
        f_map = self.extrage_feature_map(x).cpu().numpy()[0]
        c, h, w = f_map.shape
        f_map = f_map.reshape(c, h * w).T 
        
        distances = []
        for i in range(h * w):
            dist = mahalanobis(f_map[i], self.mean[:, i], self.inv_covariance[i])
            distances.append(dist)
            
        anomaly_map = np.array(distances).reshape(h, w)
        score = np.max(anomaly_map) 
        return score, anomaly_map

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'mean': self.mean, 'inv': self.inv_covariance}, f)

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.mean = data['mean']
            self.inv_covariance = data['inv']