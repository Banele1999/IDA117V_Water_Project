from torch import nn
class NeuralNet(nn.Module):
 def __init__(self,d,residual=False,dropout=.25):
  super().__init__();self.residual=residual;self.input=nn.Sequential(nn.Linear(d,64),nn.ReLU());self.hidden=nn.Sequential(nn.Linear(64,64),nn.ReLU(),nn.Dropout(dropout));self.output=nn.Linear(64,1)
 def forward(self,x):
  h=self.input(x);z=self.hidden(h);return self.output(h+z if self.residual else z).flatten()
