import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from sos import Sos
import collections
    

class Preprocess(nn.Module):
    """Preprocess images using CNN"""
    
    def __init__(self):
        """Defines parameters for module"""
        super(Preprocess, self).__init__()
        self.fc1 = nn.Linear(2,2)

    def forward(self, x):
        """Performs preprocessing to a [B,16,16,16] block"""
        return torch.ones([x.shape[0], 16, 16, 16])

class Localiser(nn.Module):
    """Output Position"""

    def __init__(self):
        """Defines parameters for module"""
        super(Localiser, self).__init__()
        self.fc1 = nn.Linear(32,4)
        
    def forward(self, x, att):
        """Outputs position and size [x, y, sx, sy]"""

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

        # MLP
        x = torch.ones([b,32])
        x = torch.sigmoid(self.fc1(x))
        return x

class Attention(nn.Module):
    """Given preproceesed image and label output attention mask"""

    def __init__(self, no_categrories):
        """Holds parameters for module"""
        super(Attention, self).__init__()
        self.fc1 = nn.Linear(2,2)

    def forward(self, x, label):
        """computes an attention mask"""
        return torch.ones([x.shape[0], 16])

class Classifier(nn.Module):
    """Given sub-image produces label"""

    def __init__(self, no_categories):
        """Holds parameters for module"""
        super(Classifier, self).__init__()
        self.no_categories = no_categories
        self.fc1 = nn.Linear(2,2)
        
    def forward(self, x):
        """Classifies sub-image x"""
        return torch.ones([x.shape[0], self.no_categories])

class NoPModel(nn.Module):
    """Network"""

    def __init__(self, no_categories):
        """Holds all models"""
        super(NoPModel, self).__init__()

        self.preprocess = Preprocess()
        self.localiser = Localiser()
        self.attention0 = Attention(no_categories)
        self.attention1 = Attention(no_categories)
        self.attention2 = Attention(no_categories)
        self.classifier = Classifier(no_categories)

    def params(self, type: int):
        if type == 0:
            params = list()
            for module in (self.preprocess, self.localiser, self.attention0, self.classifier):
                params += list(module.parameters())
            return params
        elif type == 1:
            return self.attention1.parameters()
        elif type == 2:
            return self.attention2.parameters()
        else:
            raise ValueError("Error: unknown parameters")
            
    def spatial(self, x, position):
        """Given position information return boundbox"""
        return torch.ones([*x.shape[0:2], 16, 16])

    def run(self, data):
        """Run an iteration"""
        randLabel = torch.rand([data.shape[0], 2])
        x = self.preprocess(data)
        a0 = self.attention0(x, randLabel)
        pos0 = self.localiser(x, a0)
        subx = self.spatial(data, a0)
        labels = self.classifier(subx)
        targets = torch.argmax(labels, dim=1)
        a1 = self.attention1(x, labels)
        pos1 = self.localiser(x, a1)
        a2 = self.attention2(x, None)
        pos2 = self.localiser(x, a2)
        return pos0, pos1, pos2, targets

        

class MyDataSet(Dataset):
    """A data set"""

    def __init__(self, dataset_name: str, training: bool=True):
        """Get the data"""
        super().__init__()
        self.length = 100
        self.channels = 3

    def __len__(self) -> int:
        """Return length"""
        return self.length

    def __getitem__(self, idx):
        return torch.ones([self.channels, 64, 64])

    def no_channels(self)->int:
        return self.channels



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

    def kl_loss(self, pos1, pos2):
        """KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())"""
        mu1 = pos1[:,0:2]
        sigma1 = pos1[:,2:4]
        mu2 = pos2[:,0:2]
        sigma2 = pos2[:,2:4]
        r2 = torch.div(sigma1, sigma2).pow(2)
        return -0.5 * torch.sum(r2 - 1 - r2.log() + torch.div((mu1-mu2), sigma2).pow(2))

    def choose_dataset(self, dataset_name: str):
        """Choose a dataset"""
        return MyDataSet();


    def run(self, dataset_name: str, no_epochs: int)->dict:
        """Runs the training algorithm and returns resuls"""

        # Dataset
        training_set = MyDataSet(dataset_name, True)
        testing_set = MyDataSet(dataset_name, False)
        self.channels = training_set.no_channels()

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
 #           if torch.cuda.device_count()>0:
 #               model = nn.DataParallel(model)
            model = model.to(self.device)

        # Define Optimisers
        optimiser0 = optim.SGD(model.params(0), lr=self.learning_rate, momentum=0.99)
        optimiser1 = optim.SGD(model.params(1), lr=self.learning_rate, momentum=0.99)
        optimiser2 = optim.SGD(model.params(2), lr=self.learning_rate, momentum=0.99)


        Metrics = collections.namedtuple('Metrics', ['epoch', "kl1", "kl2"])
        results = []
        for epoch in range(no_epochs):
            for i, data in enumerate(trainloader, 0):
                optimiser0.zero_grad();
                optimiser1.zero_grad();
                optimiser2.zero_grad();
                pos0, pos1, pos2, _ = model.run(data)
                loss1 = self.kl_loss(pos0, pos1)
                #loss1.backward()
                optimiser0.step()
                loss2 = self.kl_loss(pos0, pos2);
                #loss2.backward()
                optimiser1.step()
                loss = loss1-loss2
                #loss.backward()
                optimiser2.step()
                

            if epoch % 10 == 9:
                loss1 = torch.zeros([1])
                loss2 = torch.zeros([1])
                cnt = 0;
                with torch.no_grad():
                    for i, data in enumerate(trainloader, 0):
                        pos0, pos1, pos2, _ = model.run(data)
                        loss1 += torch.mean(self.kl_loss(pos0, pos1))
                        loss2 += torch.mean(self.kl_loss(pos0, pos2))
                        cnt += 1
                    res = Metrics(epoch+1, loss1.item()/cnt, loss2.item()/cnt)
                    results.append(res._asdict())
                    print("%d KL1 = %.3f KL2 = %.3f" %
                          (epoch+1, loss1/cnt, loss2/cnt))
            
        return results;
