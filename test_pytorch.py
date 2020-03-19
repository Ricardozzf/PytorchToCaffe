import sys
sys.path.insert(0,'.')
sys.path.insert(0, "/home/zouzhaofan/Work/Github/PytorchToCaffe/model/centernet_lib")
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.hub import load_state_dict_from_url
from model.centernet_lib.models.model import create_model, load_model
from model.centernet_lib.opts import opts
from model.centernet_lib.datasets.dataset_factory import get_dataset
import numpy as np



if __name__=='__main__':
    name='centernet_dcnv2'
    data = np.load('input.npy')
    data = torch.from_numpy(data)
    opt = opts().parse()
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
    Dataset = get_dataset(opt.dataset, opt.task)
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    net = create_model(opt.arch, opt.heads, opt.head_conv)
    net = load_model(net, "model_last.pth")
    device = torch.device("cuda:0")
    data = data.to(device=device).type(torch.float)
    net = net.to(device)
    net.eval()
    
    out = net(data)
    import pdb; pdb.set_trace()
    np.savetxt('out_pytorch',out[0][0,0,:4,:4].detach().cpu().numpy())
