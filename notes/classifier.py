import torch
import torch.nn as nn
import torch.nn.functional as F

NoInChannels = 3;

class Classifier(nn.Module):

	def __init__(self):
		super(Classifier, self).__init__()
		self.conv1 = nn.Conv2d(NoInChannels,16, kernel_size=5, padding=2)
		self.conv2 = nn.Conv2d(16,16, kernel_size=3, padding=1)
		self.conv3 = nn.Conv2d(16,16, kernel_size=3, padding=1)
		self.conv4 = nn.Conv2d(16,16, kernel_size=3, padding=1)
		self.fc1 = nn.Linear(256, 64)
		self.fc2 = nn.Linear(64, 32)
		self.fc3 = nn.Linear(32, 16)

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.relu(F.max_pool2d(self.conv2(x), 2))
		x = F.relu(self.conv3(x))
		x = F.relu(F.max_pool2d(self.conv4(x), 2))
		x = x.view(-1, 256)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.gumbel_softmax(self.fc3(x), hard=True)
		return x

classifier = Classifier();

batchsize = 3

x = torch.rand([batchsize, NoInChannels, 16, 16])

to = classifier.forward(x)
print(to.shape)
print(to)
