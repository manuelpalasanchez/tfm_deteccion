"""FreqNet detector"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adam, Optimizer

from models.base_model import BaseDetector
from models.model_registry import register


def _conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def _conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class _Bottleneck(nn.Module):
    """
    Bloque residual. Hace 3 convoluciones: 
    -1x1 para reducir canales
    -3x3 para procesar
    -1x1 para restaurar canales.
    inplanes: canales que llegan al bloque
    planes: canales intermedios (antes de expansion)
    downsample: conv1x1 opcional que ajusta para que se pueda sumar
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = _conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = _conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = _conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x #Se guarda la entrada para la conexion residual
        out = self.relu(self.bn1(self.conv1(x))) #con1x1 para reducir canales
        out = self.relu(self.bn2(self.conv2(out))) #con3x3 para procesar
        out = self.bn3(self.conv3(out)) #con1x1 para expandir
        if self.downsample is not None:
            identity = self.downsample(x) #ajusta dimensiones
        out += identity #suma a la salida la entrada original (conexion residual)
        return self.relu(out) #relu final


@register("freqnet")
class FreqNetDetector(BaseDetector):
    """FreqNet: detector basado en filtracion de altas frecuencias en dominio FFT
    intercalada con bloques residuales Bottleneck.
    No usa pesos preentrenados (arquitectura propia del paper).
    """

    def __init__(self):
        """
        4 etapas de procesamiento frecuencial:
        Cada una con 
        - Filtro convolucional aprendible
        - Convolucion compleja (separada en parte real e imaginaria)  

        Las etapas tienen dim diferentes:
        1: 3 canales (RGB) -> 64 canales
        2: 64 -> 64 con stride 2 (downsampling espacial)
        3: 256 -> 256 post layer1, que produce 256 canales
        4: 256 -> 256 con stride 2  
        """
        super().__init__()

        # 1 Convolucion de entrada y filtros frecuenciales aprendibles (etapa 1: 3->64 canales)
        self.weight1 = nn.Parameter(torch.randn(64, 3, 1, 1))
        self.bias1 = nn.Parameter(torch.randn(64))
        self.realconv1 = _conv1x1(64, 64)
        self.imagconv1 = _conv1x1(64, 64)

        # Etapa 2: 64->64 con stride 2 (downsampling espacial)
        self.weight2 = nn.Parameter(torch.randn(64, 64, 1, 1))
        self.bias2 = nn.Parameter(torch.randn(64))
        self.realconv2 = _conv1x1(64, 64)
        self.imagconv2 = _conv1x1(64, 64)

        # Etapa 3: post layer1, 256->256
        self.weight3 = nn.Parameter(torch.randn(256, 256, 1, 1))
        self.bias3 = nn.Parameter(torch.randn(256))
        self.realconv3 = _conv1x1(256, 256)
        self.imagconv3 = _conv1x1(256, 256)

        # Etapa 4: 256->256 con stride 2, antes de layer2
        self.weight4 = nn.Parameter(torch.randn(256, 256, 1, 1))
        self.bias4 = nn.Parameter(torch.randn(256))
        self.realconv4 = _conv1x1(256, 256)
        self.imagconv4 = _conv1x1(256, 256)


        self.inplanes = 64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(_Bottleneck, 64, blocks=3)
        self.layer2 = self._make_layer(_Bottleneck, 128, blocks=4, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 1)

        self._init_weights()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                _conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _init_weights(self):
        """Inicializa pesos de convoluciones con Kaiming (lo recomendado para ReLU)"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _hfreq_wh(self, x, scale=4):
        """Filtra y se queda con las freq altas en el dominio espacial (HxW), llevando la imagen al fft y haciendo zeroing el centro del espectro.
        Zeroing pone un cuadrado de ceros en el centro del espectro de frecuencias, 
        lo que elimina las componentes de baja frecuencia (informacion global) y deja pasar las altas frecuencias (detalles finos).
        Visualmente: Bordes muy marcados, texturas resaltadas, pero perdida de formas generales y colores.
        """
        x = torch.fft.fftshift(torch.fft.fft2(x, norm="ortho"), dim=[-2, -1])
        b, c, h, w = x.shape #Saca del tensor las dimensiones de batch, canales, altura y anchura
        x[:, :, h // 2 - h // scale: h // 2 + h // scale, 
               w // 2 - w // scale: w // 2 + w // scale] = 0.0 #zeroing 
        x = torch.fft.ifft2(torch.fft.ifftshift(x, dim=[-2, -1]), norm="ortho")#Volver al dominio espacial con ifft2 e ifftshift (inversa de fftshift)
        return F.relu(x.real, inplace=True) #Relu a la parte real 

    def _hfreq_c(self, x, scale=4):
        """Lo mismo que _hfreq_wh pero filtrando en el dominio de canales (C), aplicando fft a lo largo de la dimension de canales 
        y haciendo zeroing en el centro del espectro de canales.
        Visualmente: Resalta patrones que se repiten a lo largo de los canales (como texturas comunes en RGB) y reduce informacion global de color.
        Aqui las frecuencias se refieren a correlaciones entre canales, no espaciales. 
        El zeroing elimina patrones comunes a los canales y resalta diferencias finas entre ellos.
        """
        x = torch.fft.fftshift(torch.fft.fft(x, dim=1, norm="ortho"), dim=1)
        b, c, h, w = x.shape
        x[:, c // 2 - c // scale: c // 2 + c // scale, :, :] = 0.0
        x = torch.fft.ifft(torch.fft.ifftshift(x, dim=1), dim=1, norm="ortho")
        return F.relu(x.real, inplace=True)

    def _freq_conv(self, x, realconv, imagconv):
        """Convolucion compleja: aplica conv separadamente a parte real e imaginaria.
        Convolucion en el dominio de la frecuencia, que puede aprender a detectar patrones específicos de frecuencias altas.
        """
        x = torch.fft.fftshift(torch.fft.fft2(x, norm="ortho"), dim=[-2, -1])
        x = torch.complex(realconv(x.real), imagconv(x.imag))#Aplica convolucion a parte real e imaginaria por separado, manteniendo la naturaleza compleja del tensor.
        x = torch.fft.ifft2(torch.fft.ifftshift(x, dim=[-2, -1]), norm="ortho")
        return F.relu(x.real, inplace=True)

    def forward(self, x):
        """
        Pasa la imagen por las 4 etapas de procesamiento frecuencial intercaladas con los bloques residuales, 
        y luego por la parte final de ResNet (avgpool + fc) para clasificacion.

        El patrón es el siguiente:
        - Filtra frecuencias bajas
        - Convolucion compleja para detectar patrones en el dominio frecuencial
        - Bloque residual para combinar y refinar patrones
        - Repite con diferentes dimensiones y downsampling para extraer patrones a distintas escalas
        """
        
        x = self._hfreq_wh(x) #Pilla altas frecuencias espaciales
        x = F.relu(F.conv2d(x, self.weight1, self.bias1), inplace=True)
        x = self._hfreq_c(x) #Pilla altas frecuencias de canales
        x = self._freq_conv(x, self.realconv1, self.imagconv1)#Convolucion compleja en frecuencias
        #En este punto la imagen ha sido filtrada para resaltar detalles finos y patrones de alta frecuencia
        #  tanto espaciales como entre canales, y ha pasado por una convolucion que puede aprender a detectar patrones especificos en este dominio frecuencial.
        x = self._hfreq_wh(x)
        x = F.relu(F.conv2d(x, self.weight2, self.bias2, stride=2), inplace=True)
        x = self._hfreq_c(x)
        x = self._freq_conv(x, self.realconv2, self.imagconv2)
        #Despues de la etapa 2, la imagen ha sido filtrada y convolucionada nuevamente, 
        # con un downsampling espacial que reduce la resolucion pero mantiene los patrones de alta frecuencia detectados.
        x = self.maxpool(x)
        x = self.layer1(x)
        # Tras layer1, la imagen ha pasado por 3 bloques residuales que pueden aprender a combinar y refinar los patrones de alta frecuencia detectados en las etapas anteriores,
        # produciendo una representación más abstracta y rica en características para la clasificación final.
        x = self._hfreq_wh(x)
        x = F.relu(F.conv2d(x, self.weight3, self.bias3), inplace=True)
        x = self._freq_conv(x, self.realconv3, self.imagconv3)
        # La etapa 3 aplica filtrado y convolucion frecuencial sin downsampling, lo que permite 
        # refinar aún más los patrones de alta frecuencia en la representación de 256 canales 
        # producida por layer1.
        x = self._hfreq_wh(x)
        x = F.relu(F.conv2d(x, self.weight4, self.bias4, stride=2), inplace=True)
        x = self._freq_conv(x, self.realconv4, self.imagconv4)
        # La etapa 4 hace un último filtrado y convolucion frecuencial con downsampling,
        x = self.layer2(x)
        x = self.avgpool(x)
        x = self.fc(x.flatten(1))
        return x

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        # La normalizacion ImageNet ya se aplica en transforms.py
        return x

    def get_optimizer(self, lr: float) -> Optimizer:
        return Adam(self.parameters(), lr=lr)
