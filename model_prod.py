import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


####
# Base encoder for product construction
####

class Encoder(nn.Module):
    def __init__(self, k, h, n, **extra_kwargs):
        super(Encoder, self).__init__()
        
        self.this_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        layers = [nn.Linear(k, h), nn.SELU()]
        for _ in range(5):  
            layers.extend([nn.Linear(h, h), nn.SELU()])
        layers.append(nn.Linear(h, n))

        self._f = nn.Sequential(*layers)
        
    def power_constraint(self, codes):
        codes_mean = torch.mean(codes)
        codes_std  = torch.std(codes)
        codes_norm = (codes-codes_mean)/ codes_std
        
        return codes_norm
        
    def forward(self, inputs):
        inputs = inputs.type(torch.FloatTensor).to(self.this_device)
        x = self._f(inputs)
        return x

####
# Product construction
####

class ProductAEEncoder(nn.Module):
    def __init__(self, K, N):
        super(ProductAEEncoder, self).__init__()
        self.K = K
        self.n1, self.n2 = N
        self.encoders = nn.ModuleList()
        for k, n in zip(K, N):
            self.encoders.append(Encoder(k, 200, n))  # using m=2
            
    def forward(self, U):
        B = U.shape[0]
        U = U.reshape(B, self.K[0], self.K[1])
        # Step 1: Apply the first encoder to each length-k1 row of U.
        U1 = U.view(-1, self.encoders[0]._f[0].in_features)  # Reshape to match encoder input
        U1 = self.encoders[0](U1).view(U.shape[0], self.K[1], -1)  # Encoder output reshaped into (batch_size, k2, n1)
    
        # Step 2: Apply the second encoder to each length-k2 vector in the second dimension of U1.
        U2 = U1.transpose(1, 2).contiguous()  # Transpose to bring the k2 dimension to the last dimension, and make it contiguous
        U2 = U2.view(-1, self.encoders[1]._f[0].in_features)  # Reshape to match encoder input
        U2 = self.encoders[1](U2).view(U1.shape[0], -1, U1.shape[1])  # Encoder output reshaped into (batch_size, n2, n1)
    
        # Step 3:
        U2_flat = U2.view(U2.shape[0], -1)  # Flatten U2 into shape (batch_size, -1)
        C = self.encoders[0].power_constraint(U2_flat)  # Apply power normalization
        return C
    
####
# Decoder base
####

class Decoder(nn.Module):
    def __init__(self, k, h, n, **extra_kwargs):
        super(Decoder, self).__init__()        
        layers = [nn.Linear(n, h), nn.SELU()]
        for _ in range(5): 
            layers.extend([nn.Linear(h, h), nn.SELU()])
        layers.append(nn.Linear(h, k))

        self._f = nn.Sequential(*layers)
        
    def forward(self, inputs):
        x = self._f(inputs)
        return x
    
class Decoder_last(nn.Module):
    def __init__(self, k, h, n, **extra_kwargs):
        super(Decoder_last, self).__init__()        
        layers = [nn.Linear(n, h), nn.SELU()]
        for _ in range(7):
            layers.extend([nn.Linear(h, h), nn.SELU()])
        layers.append(nn.Linear(h, k))

        self._f = nn.Sequential(*layers)
        
    def forward(self, inputs):
        x = self._f(inputs)
        return x    
    
####
# Decoder construction
####
    
class ProdDecoder(nn.Module):
    def __init__(self, I, K, N, **extra_kwargs):
        super(ProdDecoder, self).__init__()

        self.I = I
        self.k1, self.k2 = K
        self.n1, self.n2 = N
        self.F = 3
        M = 250

        self.decoders_1 = nn.ModuleList()
        self.decoders_2 = nn.ModuleList()

        for i in range(I):
            if i == 0:
                # The first decoder works with original input sizes n1 and n2
                self.decoders_1.append(Decoder(self.F*self.n1, M, (1+self.F)*self.n1, **extra_kwargs))
                self.decoders_2.append(Decoder(self.F*self.n2, M, self.n2, **extra_kwargs))
            elif i < I - 1:
                # Subsequent decoders (except the last one) work with input sizes 2*n1 and 2*n2
                self.decoders_1.append(Decoder(self.F*self.n1, M, (1+self.F)*self.n1, **extra_kwargs))
                self.decoders_2.append(Decoder(self.F*self.n2, M, (1+self.F)*self.n2, **extra_kwargs))
            else:
                # The last decoder reverts the encoding operation by reducing the lengths from 2*n1 and 2*n2 to k1 and k2
                self.decoders_1.append(Decoder_last(self.k1, M, self.F*self.n1, **extra_kwargs))
                self.decoders_2.append(Decoder_last(self.F*self.k2, M, (1+self.F)*self.n2, **extra_kwargs))
                
    def forward(self, Y):
        B = Y.size(0)
        Y = Y.view(B, self.n1, self.n2)
    
        if self.I == 1:
            Yin2_2 = Y
        else:
            for i in range(self.I-1):
                if i == 0:
                    Y2 = self.decoders_2[i](Y).view(B, self.F*self.n1, self.n2)
                else:
                    Yout2 = self.decoders_2[i](Yin2_2)
                    Y2 = (Yout2 - Yin2_1).view(B, self.F*self.n1, self.n2)

                Yin1 = torch.cat([Y, Y2], dim=1).permute(0, 2, 1)
                Y1 = self.decoders_1[i](Yin1).permute(0, 2, 1)
                Yin2_1 = (Y1 - Y2).reshape(B, self.n1, self.F*self.n2)
                Yin2_2 = torch.cat([Y, Yin2_1], dim=2)

        Y2 = self.decoders_2[-1](Yin2_2).view(B, self.F*self.n1, self.k2)
        Y1 = self.decoders_1[-1](Y2.permute(0, 2, 1))
        U_hat = Y1.view(B, self.k1*self.k2)
        m = nn.Sigmoid()
        return m(U_hat)
