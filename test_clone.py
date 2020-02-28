import torch.nn as nn
import torch
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 6, kernel_size=1, stride=1, bias=False)
        self.conv2 = nn.Conv2d(6, 6, kernel_size=1, stride=1, bias=False)
        self.conv3 = nn.Conv2d(6, 6, kernel_size=1, stride=1, bias=False)
        
    def forward(self,input):
        input = F.relu(self.conv1(input))
        input2 = input
        
        output = F.relu(self.conv2(input))
        output2 = F.relu(self.conv3(input2))
        
        return output, output2
        
class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 6, kernel_size=1, stride=1, bias=False)
        self.conv2 = nn.Conv2d(6, 6, kernel_size=1, stride=1, bias=False)
        self.conv3 = nn.Conv2d(6, 6, kernel_size=1, stride=1, bias=False)
        
    def forward(self,input):
        input = F.relu(self.conv1(input))
        input2 = input.clone()
        
        output = F.relu(self.conv2(input))
        output2 = F.relu(self.conv3(input2))
        
        return output, output2

if __name__ == "__main__":
    model1 = Net()
    model2 = Net2()
    model2.load_state_dict(model1.state_dict())

    x = torch.randn(1, 3, 24, 24)

    outputs1 = model1(x)
    outputs2 = model2(x)

    # Compare outputs
    print((outputs1[0] == outputs2[0]).all())
    print((outputs1[1] == outputs2[1]).all())

    # Compare gradients
    outputs1[0].mean().backward(retain_graph=True)
    outputs1[1].mean().backward()
    outputs2[0].mean().backward(retain_graph=True)
    outputs2[1].mean().backward()

    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        print((p1.grad == p2.grad).all())