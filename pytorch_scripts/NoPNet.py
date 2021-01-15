import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from sos import Sos
import collections
from mydataset import load_data, DataSetType
import pytest

class Preprocess(nn.Module):
    """Preprocess images using CNN"""
    
    def __init__(self):
        """Defines parameters for module"""
        super(Preprocess, self).__init__()
        # CNN0
        self.conv1 = nn.Conv2d(3,16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(16,16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16,16, kernel_size=3, padding=1)
        # CNN1
        self.conv4 = nn.Conv2d(16,16, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(16,16, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(16,16, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 32)

    def forward(self, x: "Tensor") -> ("Tensor", "Tensor"):
        """Performs preprocessing 
        Input  x = [B, 3, 64, 64]
        Output x = [B, 16, 16, 16] block
        Output y = [B, 32]"""

        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)

        # Attention block CNN1
        y = F.max_pool2d(F.relu(self.conv4(x)), 2)
        y = F.max_pool2d(F.relu(self.conv5(y)), 2)
        y = F.relu(self.conv6(y))
        y = y.view(-1, 256)
        y = F.relu(self.fc1(y))
        y = self.fc2(y)
        return x, y


class Attention(nn.Module):
    """Given preproceesed image and label output attention mask"""

    def __init__(self, no_categories):
        """Holds parameters for module"""
        super(Attention, self).__init__()
        if no_categories is not None:           
            self.fc0 = nn.Linear(no_categories, 32)
            self.fc1 = nn.Linear(64, 64)
        else:
            self.fc01 = nn.Linear(32,64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)

    def forward(self, x: "Tensor", label: "Tensor") -> "Tensor":
        """computes an attention mask
        Input  x = [B,32] # image attention
        Input  label = [B,no_categories]
        Output x = [N,16] % channel normalisation"""

        if label is not None:
            label = F.relu(self.fc0(label))
            x = torch.cat([label, x], dim=1)
            x = F.relu(self.fc1(x))
        else:
            x = F.relu(self.fc01(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LocationModule(nn.Module):
    """Output Position"""

    def __init__(self):
        """Defines parameters for module"""
        super(LocationModule, self).__init__()
        self.conv1 = nn.Conv2d(18,16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16,16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16,16, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(16,16, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(16,16, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(256, 32)
        self.fc2 = nn.Linear(32, 4)

        
    def forward(self, x: "Tensor", att: "Tensor") -> "Tensor":
        """Locates an object in a image
        Input   x = [B, 16, 16, 16]
        Input att = [B, 16]
        Output  x = [x, y, sx, sy]"""

        # multiplicative attention
        x = torch.einsum("bcwh,bc->bcwh", [x,att])

        # add coordinates
        width = x.shape[-1]
        b = x.shape[0]
        ones = torch.ones(width)
        seq = torch.linspace(0,1,width)
        colCoord = torch.einsum("a,b->ab", [ones,seq]).repeat(b,1,1,1)
        rowCoord = torch.einsum("a,b->ab", [seq,ones]).repeat(b,1,1,1)
        x = torch.cat((x,colCoord,rowCoord), dim=1)

        # CNN
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(self.conv4(x), 2)
        x = self.conv5(x)

        # MLP
        x = x.view(-1, 256)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x


class Classifier(nn.Module):
    """Given sub-image produces label"""

    def __init__(self, no_categories, no_in_channels=3):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(no_in_channels,16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(16,16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16,16, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(16,16, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, no_categories)

    def forward(self, x):
        """Classifies sub-image x"""
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(self.conv3(x))
        x = F.relu(F.max_pool2d(self.conv4(x), 2))
        x = x.view(-1, 256)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.gumbel_softmax(self.fc3(x), hard=True)
        return x


class NoPModel(nn.Module):
    """Network"""

    def __init__(self, no_categories):
        """Holds all models"""
        super(NoPModel, self).__init__()

        self.no_categories = no_categories
        self.preprocess = Preprocess()
        self.localiser = LocationModule()
        self.attention0 = Attention(no_categories)
        self.attention1 = Attention(no_categories)
        self.attention2 = Attention(None)
        self.classifier = Classifier(no_categories)

    def parameters(self):
        params0 = list()
        for module in (self.preprocess, self.localiser, self.attention0, self.classifier):
            params0 += list(module.parameters())
        return params0, self.attention1.parameters(), self.attention2.parameters()

            
    def spatial(self, x, position):
        """Given position information return boundbox"""
        return torch.ones([*x.shape[0:2], 16, 16])

    def forward(self, data):
        """Run an iteration"""
        randLabel = torch.rand([data.shape[0], self.no_categories])
        x, y = self.preprocess(data)
        a0 = self.attention0(y, randLabel)
        pos0 = self.localiser(x, a0)
        subx = self.spatial(data, a0)
        labels = self.classifier(subx)
        targets = torch.argmax(labels, dim=1)
        a1 = self.attention1(y, labels)
        pos1 = self.localiser(x, a1)
        a2 = self.attention2(y, None)
        pos2 = self.localiser(x, a2)
        return pos0, pos1, pos2, targets

        
def kl_loss(pos1, pos2):
    """KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())"""
    mu1 = pos1[:,0:2]
    sigma1 = pos1[:,2:4]
    mu2 = pos2[:,0:2]
    sigma2 = pos2[:,2:4]
    r2 = torch.div(sigma1, sigma2).pow(2)
    return 0.5 * torch.sum(r2 - 1 - r2.log() + torch.div((mu1-mu2), sigma2).pow(2))


class NoPTrain():
    def __init__(self, args, use_cuda: bool, device):
        """Trains the Naming of Parts network"""
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.no_categories = args.no_categories
        self.channels = 3
        self.num_workers = args.num_workers
        self.use_cuda = use_cuda
        self.device = device
        if self.use_cuda:
            self.batch_size *= torch.cuda.device_count()


    def run(self, dataset_name: str, no_epochs: int)->dict:
        """Runs the training algorithm and returns resuls"""

        # Dataset
        training_set =load_data(dataset_name, DataSetType.Train)
        testing_set = load_data(dataset_name, DataSetType.Test)
        self.channels = training_set[0].shape[1]

        trainloader = DataLoader(training_set,
                                 batch_size=self.batch_size,
                                 shuffle=True,
                                 num_workers=self.num_workers,
                                 pin_memory=self.use_cuda)
        
        testloader = DataLoader(training_set,
                                batch_size=self.batch_size, shuffle=False,
                                num_workers=self.num_workers,
                                pin_memory=self.use_cuda)
        
        # Define Networks
        model = NoPModel(self.no_categories)
        if self.use_cuda:
            if torch.cuda.device_count()>0:
                model = nn.DataParallel(model)
            model = model.to(self.device)

        # Define Optimisers
        params0, params1, params2 = model.parameters();
        optimiser0 = optim.SGD(params0, lr=self.learning_rate, momentum=0.99)
        optimiser1 = optim.SGD(params1, lr=self.learning_rate, momentum=0.99)
        optimiser2 = optim.SGD(params2, lr=self.learning_rate, momentum=0.99)


        Metrics = collections.namedtuple('Metrics', ['epoch', "kl1", "kl2"])
        results = []
        for epoch in range(no_epochs):
            for i, data in enumerate(trainloader):
                optimiser0.zero_grad();
                optimiser1.zero_grad();
                optimiser2.zero_grad();
                pos0, pos1, pos2, _ = model(data)
                loss1 = kl_loss(pos0, pos1)
                loss1.backward(retain_graph=True)
                optimiser0.step()
                loss2 = kl_loss(pos0, pos2);
                loss2.backward(retain_graph=True)
                optimiser1.step()
                loss = loss1-loss2
                loss.backward()
                optimiser2.step()
                

            if epoch % 10 == 9:
                loss1 = torch.zeros([1])
                loss2 = torch.zeros([1])
                cnt = 0;
                with torch.no_grad():
                    for i, data in enumerate(trainloader):
                        pos0, pos1, pos2, _ = model(data)
                        loss1 += torch.mean(kl_loss(pos0, pos1))
                        loss2 += torch.mean(kl_loss(pos0, pos2))
                        cnt += 1
                    res = Metrics(epoch+1, loss1.item()/cnt, loss2.item()/cnt)
                    results.append(res._asdict())
                    print("%d KL1 = %.3f KL2 = %.3f" %
                          (epoch+1, loss1/cnt, loss2/cnt))
            
        return results;

