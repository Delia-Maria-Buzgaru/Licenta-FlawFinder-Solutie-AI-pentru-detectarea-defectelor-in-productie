import torch
import torch.nn as nn
import numpy as np

class Autoencoder(nn.Module):
    def __init__(self, width, height, depth, filtre=(32, 64, 128, 256), DimLatenta=1024):
        super(Autoencoder, self).__init__()
        self.filtre = filtre
        self.DimLatenta = DimLatenta
        self.depth = depth

        #Encoder-
        layers_e = []
        canal_curent = depth
        for f in filtre:
            layers_e.append(nn.Conv2d(canal_curent, f, kernel_size=3, stride=2, padding=1))
            layers_e.append(nn.LeakyReLU(0.2))
            layers_e.append(nn.BatchNorm2d(f))
            canal_curent = f
        
        self.formaOriginala = (filtre[-1], height // (2**len(filtre)), width // (2**len(filtre)))
        total_trasaturi = np.prod(self.formaOriginala)

        # Defineste encoder-ul ca un singur bloc unitar
        self.encoder = nn.Sequential(
            nn.Sequential(*layers_e),
            nn.Flatten(),
            nn.Linear(total_trasaturi, DimLatenta)
        )

        #Decoder
        self.decoder_linear = nn.Linear(DimLatenta, total_trasaturi)
        
        layers_d = []
        filtre_inverse = list(filtre[::-1])
        for i in range(len(filtre_inverse) - 1):
            layers_d.append(nn.ConvTranspose2d(filtre_inverse[i], filtre_inverse[i+1], kernel_size=3, stride=2, padding=1, output_padding=1))
            layers_d.append(nn.LeakyReLU(0.2))
            layers_d.append(nn.BatchNorm2d(filtre_inverse[i+1]))

        layers_d.append(nn.ConvTranspose2d(filtre_inverse[-1], filtre_inverse[-1], kernel_size=3, stride=2, padding=1, output_padding=1))
        layers_d.append(nn.LeakyReLU(0.2))
            
        self.decoder_conv = nn.Sequential(*layers_d)

        self.strat_final = nn.Sequential(
            nn.Conv2d(filtre_inverse[-1], depth, kernel_size=3, padding=1),
            nn.Sigmoid() # forteaza pixelii sa fie intre 0 si 1
        )

    def decoder(self, z):
        x = self.decoder_linear(z)
        x = x.view(-1, *self.formaOriginala)
        x = self.decoder_conv(x)
        x = self.strat_final(x)
        return x

    def forward(self, x):
        cod_latent = self.encoder(x)
        reconstructie = self.decoder(cod_latent)
        return reconstructie