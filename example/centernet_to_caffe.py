import sys
sys.path.insert(0,'.')
sys.path.insert(0, "/home/zouzhaofan/Work/Github/PytorchToCaffe/model/centernet_lib")
import torch
import torch.nn as nn
from torch.autograd import Variable
#from torchvision.models.alexnet import alexnet
from torch.hub import load_state_dict_from_url
import pytorch_to_caffe
from model.centernet_lib.models.model import create_model, load_model
from model.centernet_lib.opts import opts
from model.centernet_lib.datasets.dataset_factory import get_dataset
import numpy as np


if __name__=='__main__':
    name='centernet_dcnv2'
    opt = opts().parse()
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
    Dataset = get_dataset(opt.dataset, opt.task)
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    net = create_model(opt.arch, opt.heads, opt.head_conv)
    net = load_model(net, "model_last.pth")
    device = torch.device("cuda")
    net = net.to(device)
    net.eval()
    
    #with open("centernet.txt", "w", encoding="utf-8") as f:
    #    print(net, file=f)
    input = np.load("centernet_data.npy")
    input = torch.from_numpy(input).type(torch.float32).to(device)
    output = net(input)
    import pdb; pdb.set_trace()
    '''
    pytorch_to_caffe.trans_net(net,input,name)
    
    pytorch_to_caffe.save_prototxt('{}.prototxt'.format(name))
    pytorch_to_caffe.save_caffemodel('{}.caffemodel'.format(name))
    '''