import sys
sys.path.insert(0,'.')
import torch
import torch.nn as nn
from torch.autograd import Variable
#from torchvision.models.alexnet import alexnet
from torch.hub import load_state_dict_from_url
import pytorch_to_caffe
import numpy as np
from trd.Dcn.modules.deform_conv import DeformConvPack as Dcnv2


__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.dcn = Dcnv2(3,3,3,1,1)
        for layer in self.features:
            if isinstance(layer,nn.Conv2d):
                torch.nn.init.kaiming_uniform_(layer.weight)
                layer.bias.data.zero_()
        

    def forward(self, x):
        x = self.features(x)
        
        x = self.dcn(x)
        return x


def alexnet(pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def register(module):
    module.register_forward_hook(module_hook)

def module_hook(module, input, output):
    #print("input:{}".format(id(input)))
    #print("output:{}".format(id(output)))
    print(output[0,0,:,:])

def _Inplace(net):
    for module in net.modules():
        if isinstance(module,torch.nn.ReLU):
            module.inplace = False
    return net


if __name__=='__main__':
    name='alexnet'
    net=AlexNet()
    net.eval()
    net.cuda()
    torch.save(net,"alex.pt")
    #net = torch.load("alex.pt")
    #net.dcn.conv_offset.register_forward_hook(module_hook)
    

    #net = _Inplace(net)
    #net.apply(register)
    #input=Variable(torch.ones([1,3,226,226]))
    
    input = np.load('input_alex.npy')
    input_tensor = torch.from_numpy(input).type(torch.float32).to("cuda")
    out = net(input_tensor)
    
    #np.savetxt("tras_tensor",out[0,0,:,:].cpu().detach().numpy())

    
    pytorch_to_caffe.trans_net(net,input_tensor,name)
    pytorch_to_caffe.save_prototxt('{}.prototxt'.format(name))
    pytorch_to_caffe.save_caffemodel('{}.caffemodel'.format(name))
    