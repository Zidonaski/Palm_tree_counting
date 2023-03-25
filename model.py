import torchvision.models as models
from torch import nn

class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    efficientnet_b3 = models.efficientnet_b3(weights="IMAGENET1K_V1")
    print("loading pretrained weights")
    num_features=efficientnet_b3.classifier[1].in_features
    efficientnet_b3.classifier[1]=nn.Linear(num_features,1)
    nn.init.xavier_uniform_(efficientnet_b3.classifier[1].weight)
    self.efficientnet_b3=efficientnet_b3
  def forward(self, x):
    return self.efficientnet_b3(x)
  def freeze(self):
    for p in self.efficientnet_b3.features.parameters():
        p.requires_grad=False
  def unfreeze(self):
    for p in self.efficientnet_b3.parameters():
        p.requires_grad=True