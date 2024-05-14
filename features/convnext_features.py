import torch
import torch.nn as nn
from torchvision import models 
import torchvision
import numpy as np

def replace_convlayers_convnext(model, threshold):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_convlayers_convnext(module, threshold)
        if isinstance(module, nn.Conv2d):
            if module.stride[0] == 2:
                if module.in_channels > threshold: #replace bigger strides to reduce receptive field, skip some 2x2 layers. >100 gives output size (26, 26). >300 gives (13, 13)
                    module.stride = tuple(s//2 for s in module.stride)
                    
    return model

def convnext_tiny_26_features(pretrained=False, **kwargs):
    model = models.convnext_tiny(pretrained=pretrained, weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
    with torch.no_grad():
        model.avgpool = nn.Identity()
        model.classifier = nn.Identity()    
        model = replace_convlayers_convnext(model, 100) 
    
    return model

def convnext_tiny_13_features(pretrained=False, **kwargs):
    model = models.convnext_tiny(pretrained=pretrained, weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
    with torch.no_grad():
        model.avgpool = nn.Identity()
        model.classifier = nn.Identity()    
        model = replace_convlayers_convnext(model, 300) 
    
    return model

def convnext_tiny_7_features(pretrained=False, **kwargs):
    model = models.convnext_tiny(pretrained=pretrained, weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
    with torch.no_grad():
        model.avgpool = nn.Identity()
        model.classifier = nn.Identity()    
    
    return model

class BasicGaussianMultiplierConv2D(nn.Conv2d):
    def __init__(self, conv_layer, sigma=1.0, factor=50, device='cuda'):
        super(BasicGaussianMultiplierConv2D, self).__init__(in_channels=conv_layer.in_channels, \
                                         out_channels=conv_layer.out_channels, \
                                         kernel_size=conv_layer.kernel_size, \
                                         stride=conv_layer.stride, \
                                         padding=conv_layer.padding, \
                                         dilation=conv_layer.dilation, \
                                         groups=conv_layer.groups, \
                                         bias=conv_layer.bias is not None, \
                                         padding_mode=conv_layer.padding_mode)
        
        self.weight.data = conv_layer.weight.data.clone()
        if conv_layer.bias is not None:
            self.bias.data = conv_layer.bias.data.clone()
        # self.gaussian_multiplier = self.generate_gaussian_kernel(size=self.weight.shape[-1], sigma=sigma).to(device)
        # self.gaussian_multiplier.requires_grad = False
        self.register_buffer('gaussian_multiplier', \
                            self.generate_gaussian_kernel(size=self.weight.shape[-1], sigma=sigma).to(device))
        self.factor = factor

    def generate_gaussian_kernel(self, size, sigma=1.0):
        kernel = np.fromfunction(
            lambda x, y: (1 / (2 * np.pi * sigma**2)) * np.exp(-((x - (size-1)/2)**2 + (y - (size-1)/2)**2) / (2 * sigma**2)),
            (size, size)
        )
        kernel = torch.Tensor(kernel)
        kernel /= kernel.sum()
        kernel = kernel
        kernel = kernel.expand(1, 1, size, size)
        return kernel

    def forward(self, input):
        if self.bias is not None:
            bias = self.bias.data
        else:
            bias = None
        gaussian_multiplied_kernel = self.weight.data * self.gaussian_multiplier * self.factor
        return self._conv_forward(input, gaussian_multiplied_kernel, bias)

def replace_CNBlock_conv_layer_with_gaussian_multiplier(model: torchvision.models.convnext.CNBlock, sigma=1.0, factor=50, device='cuda'):
    first_sequential_inside_cnblock = list(model.children())[0]
    cnblock_7x7_layer = first_sequential_inside_cnblock[0]
    first_sequential_inside_cnblock[0] = BasicGaussianMultiplierConv2D(cnblock_7x7_layer, sigma=sigma, factor=factor, device=device)
    
def replace_convnext_convlayers_with_gaussian_multipliers(model, sigma=1.0, factor=50, device='cuda'):
    for n, module in model.named_children():
        if isinstance(module, torchvision.models.convnext.CNBlock):
            replace_CNBlock_conv_layer_with_gaussian_multiplier(module, sigma=sigma, factor=factor, device=device)
        elif len(list(module.children())) > 0:
            replace_convlayers_with_gaussian_convnext(module, sigma=sigma, factor=factor, device=device)
    return model

def apply_gaussian_multiplier_to_convnext_stage(original_model, stage, sigma=1.0, factor=50, device='cuda'):
    if stage == 1:
        print('stage', stage)
        for child in list(original_model.features.children())[:2]:
            replace_convnext_convlayers_with_gaussian_multipliers(child, sigma=sigma, factor=factor, device=device)
    elif stage == 2:
        print('stage', stage)
        for child in list(original_model.features.children())[2:4]:
            replace_convnext_convlayers_with_gaussian_multipliers(child, sigma=sigma, factor=factor, device=device)
    elif stage == 3:
        print('stage', stage)
        for child in list(original_model.features.children())[4:6]:
            replace_convnext_convlayers_with_gaussian_multipliers(child, sigma=sigma, factor=factor, device=device)
    elif stage == 4:
        print('stage', stage)
        for child in list(original_model.features.children())[6:8]:
            replace_convnext_convlayers_with_gaussian_multipliers(child, sigma=sigma, factor=factor, device=device)
            
    return original_model