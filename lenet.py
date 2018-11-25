from collections import OrderedDict

from torch import nn

class Module(nn.Module):
  """Extend nn.Module with function to count number of trainable parameters.
  """

  def num_trainable_parameters(self):
    def prod(a):
      total = 1
      for x in a:
        total *= x
      return total
    return sum(prod(p.size()) for p in self.parameters() if p.requires_grad)


class LeNet(Module):
  """
  LeNet-5 from LeCun.
  Args:
    indim (int) or (tuple(int, int)): input dimension (height, width)
    out_features (int): number of output neurons in final layer
  """

  def __init__(self, indim, out_features=10):
    super(LeNet, self).__init__()
    if isinstance(indim, int):
      indim = (indim, indim)
    self.conv_layers = nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(1, 6, 5, padding=2)),
        ('relu1', nn.ReLU()),
        ('maxpool1', nn.MaxPool2d(2)),
        ('conv2', nn.Conv2d(6, 16, 5)),
        ('relu2', nn.ReLU()),
        ('maxpool2', nn.MaxPool2d(2)),
    ]))
    h = (indim[0] // 2 - 4) // 2  # output feature map size of conv block
    w = (indim[1] // 2 - 4) // 2
    self.fc_layers = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(16 * w * h, 120)),
        ('relu3', nn.ReLU()),
        ('fc2', nn.Linear(120, 84)),
        ('relu4', nn.ReLU()),
        ('fc3', nn.Linear(84, out_features, bias=False))
    ]))

  def forward(self, x):
    x = self.conv_layers(x)
    x = x.view(x.size(0), -1)
    x = self.fc_layers(x)
    return x
