import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

class PatchCore(nn.Module):
    
    def __init__(self, backbone="resnet18", flow_steps=8):
        super(PatchCore, self).__init__()
        
    
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3
        )
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor.eval()

        self.adaptor = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, 256),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, 256),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=1, bias=True),
        )

    def forward(self, x):
        self.feature_extractor.eval()
        with torch.no_grad():
            features = self.feature_extractor(x)  
        
        z = self.adaptor(features) 
        return z, features

    def compute_loss(self, z):
        return 0.5 * torch.mean(z ** 2)


class PatchCoreDetector:
    def __init__(self, model_path=None, device='cpu'):
        self.device = device
        self.model = PatchCore().to(device)
        self.train_mean = None
        self.train_std = None
        
        if model_path:
            self.load(model_path)

    def train_model(self, dataloader, epochs=50, lr=2e-4):
        optimizer = torch.optim.AdamW(
            self.model.adaptor.parameters(), 
            lr=lr, 
            weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )
        
        self.model.train()
        self.model.feature_extractor.eval()

        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                z, _ = self.model(batch)
                loss = self.model.compute_loss(z)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(
                    self.model.adaptor.parameters(), 
                    max_norm=1.0
                )
                
                optimizer.step()
                total_loss += loss.item()
            
            scheduler.step()
            avg_loss = total_loss / len(dataloader)
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

        print("\nCalibrare statistici pe datele de training...")
        self._calibreaza_statistici(dataloader)

    def _calibreaza_statistici(self, dataloader):
        """Calculează media și std scorurilor pe imagini normale.
        Folosit pentru a converti scorul brut în z-score la inferență."""
        self.model.eval()
        scoruri = []
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                z, _ = self.model(batch)
                anomaly_map = torch.mean(z**2, dim=1)
                scor_per_imagine = anomaly_map.mean(dim=(1, 2))
                scoruri.extend(scor_per_imagine.cpu().numpy().tolist())

        self.train_mean = float(np.mean(scoruri))
        self.train_std = float(np.std(scoruri)) + 1e-8
        print(f"  Scor mediu pe date normale: {self.train_mean:.6f} ± {self.train_std:.6f}")
        print(f"  Prag sugerat (mean + 3*std): {self.train_mean + 3*self.train_std:.6f}")
        print(f"  In z-score: prag recomandat = 3.0 in interfata.py")

    def predict(self, input_tensor):
        self.model.eval()
        with torch.no_grad():
            z, _ = self.model(input_tensor)
            
            anomaly_map = torch.mean(z**2, dim=1).unsqueeze(1)
            anomaly_map = F.interpolate(
                anomaly_map, size=(256, 256), 
                mode='bilinear', align_corners=False
            )
            anomaly_map_np = anomaly_map.squeeze().cpu().numpy()

            scor_brut = float(np.mean(anomaly_map_np))

            if self.train_mean is not None:
                scor_final = (scor_brut - self.train_mean) / self.train_std
            else:
                scor_final = scor_brut
            
            return scor_final, [anomaly_map_np]

    def save(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'train_mean': self.train_mean,
            'train_std': self.train_std,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.train_mean = checkpoint.get('train_mean', None)
            self.train_std = checkpoint.get('train_std', None)
            if self.train_mean is not None:
                print(f"Statistici incarcate: mean={self.train_mean:.6f}, std={self.train_std:.6f}")
        else:
            self.model.load_state_dict(checkpoint)
            print("ATENTIE: Model vechi fara statistici. Scorul va fi brut.")