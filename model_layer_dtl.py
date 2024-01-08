import torch
import torchvision.models as models

# pretrained model
model = models.alexnet(pretrained=True)
input_tensor  = torch.randn(1,3,224,224)

def print_layer_dtl(model, in_tensor):
    for name, layer in model.named_modules():
        # conf_ifmap, conf_filt, conf_ofmap
        labels = ['conf_', 'height', 'width', 'n_channels', 'n_maps', 'n_zero', 'bitwidth']
        conf_ifmap = dict.fromkeys(labels)
        conf_ofmap = dict.fromkeys(labels)
        conf_filt = dict.fromkeys(labels)
        conf_ifmap['conf_'], conf_ofmap['conf_'], conf_filt['conf_'] = 'ifmap', 'ofmap', 'filt'
        conf_ifmap['n_zero'], conf_ofmap['n_zero'], conf_filt['n_zero'] = 0,0,0
        conf_ifmap['bitwidth'], conf_ofmap['bitwidth'], conf_filt['bitwidth'] = 32,32,32
        conf_ifmap['height'], conf_ifmap['width'], conf_ifmap['n_channels'], conf_ifmap['n_maps'] = in_tensor.shape[2], in_tensor.shape[3], layer.in_channels, layer.out_channels
        conf_filt['height'], conf_filt['width'], conf_filt['n_channels'], conf_filt['n_maps'] = layer.kernel_size[0], layer.kernel_size[1], layer.in_channels, layer.out_channels
        dict_list = [conf_ifmap, conf_ofmap, conf_filt]
        for d in dict_list:
            if d['conf_'] == 'ifmap':
                d['height'] = in_tensor.shape[2]
                d['width'] = in_tensor.shape[3]
                d['n_channels'] = layer.in_channels
                d['n_maps'] = layer.out_channels
        