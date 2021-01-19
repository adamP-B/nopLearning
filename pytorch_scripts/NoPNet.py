import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from sos import Sos
import collections
from mydataset import load_data, DataSetType
import pytest
import uuid


class CNNBlock(nn.Module):
    """Two CNNs with ReLUs and maxpooling"""

    def __init__(self, in_channels: int, conv_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, conv_channels,
                               kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(conv_channels, conv_channels,
                               kernel_size=3, padding=1)

    def forward(self, x: "Tensor")->"Tensor":
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        return x
        

class MLP(nn.Module):
    """Flattens Input and prodces an MLP with linear outpur"""
    
    def __init__(self, layerSizes: list):
        super().__init__()
        self.layerSizes = layerSizes
        self.layers = nn.ModuleList()
        for n1, n2 in zip(layerSizes, layerSizes[1:]):
            self.layers.append(nn.Linear(n1, n2))

    def forward(self, x: "Tensor")->"Tensor":
        x = x.view(-1,self.layerSizes[0])
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)
        return x

class Preprocess(nn.Module):
    """Preprocess images using CNN"""
    
    def __init__(self):
        """Defines parameters for module"""
        super().__init__()

        # CNN0
        self.block1 = CNNBlock(3,16)  # Bx3x64x64  -> Bx16x32x32
        self.block2 = CNNBlock(16,16) # Bx16x32x32 -> Bx16x16x16

        # CNN1
        self.block3 = CNNBlock(16, 16)   # Bx16x16x16 -> Bx16x8x8
        self.block4 = CNNBlock(16, 16)   # Bx16x8x8   -> Bx16x4x4
        self.mlp = MLP([16*4*4, 64, 32]) # Bx16x4x4   -> Bx32

    def forward(self, x: "Tensor") -> ("Tensor", "Tensor"):
        """Performs preprocessing 
        Input  x = [B, 3, 64, 64]
        Output x = [B, 16, 16, 16] block
        Output y = [B, 32]"""

        x = self.block1(x)
        x = self.block2(x)

        # Attention block CNN1
        y = self.block3(x)
        y = self.block4(y)
        y = self.mlp(y)
        
        return x, y


class Attention(nn.Module):
    """Given preproceesed image and label output attention mask"""

    def __init__(self, no_categories):
        """Holds parameters for module"""
        super().__init__()
        self.mlp0 = MLP([no_categories, 32])
        self.mlp1 = MLP([64, 32, 16])


    def forward(self, x0: "Tensor", label: "Tensor") -> "Tensor":
        """computes an attention mask
        Input  x = [B,32] # image attention
        Input  label = [B,no_categories]
        Output x = [N,16] % channel normalisation"""

        label = self.mlp0(label)
        x = torch.cat([label, x0], dim=1)
    
        x = self.mlp1(x)

        return x
    
class UnlabelledAttention(nn.Module):
    """Given preproceesed image and label output attention mask"""

    def __init__(self):
        """Holds parameters for module"""
        super().__init__()
        self.mlp1 = MLP([32, 32, 16])


    def forward(self, x: "Tensor") -> "Tensor":
        """computes an attention mask
        Input  x = [B,32] # image attention
        Input  label = [B,no_categories]
        Output x = [N,16] % channel normalisation"""

        x = self.mlp1(x)

        return x


class LocationModule(nn.Module):
    """Output Position"""

    def __init__(self, device):
        """Defines parameters for module"""
        super().__init__()
        self.block1 = CNNBlock(18, 16)  # Bx18x16x16 -> Bx16x8x8
        self.block2 = CNNBlock(16, 16)  # Bx16x8x8   -> Bx16x4x4
        self.mlp1 = MLP([16*4*4, 32, 4])
        self.device = device

        
    def forward(self, x: "Tensor", att: "Tensor") -> "Tensor":
        """Locates an object in a image
        Input   x = [B, 16, 16, 16]
        Input att = [B, 16]
        Output  x = [x, y, sx, sy]"""

        # multiplicative attention
        x = torch.einsum("bcwh,bc->bcwh", [x,att])

        # add coordinates
        b, _, _, width = x.shape
        ones = torch.ones(width, device=self.device)
        seq = torch.linspace(0,1,width, device=self.device)
        colCoord = torch.einsum("a,b->ab", [ones,seq]).repeat(b,1,1,1)
        rowCoord = torch.einsum("a,b->ab", [seq,ones]).repeat(b,1,1,1)
        x = torch.cat((x,colCoord,rowCoord), dim=1)

        # CNN + MLP
        x = self.block1(x)
        x = self.block2(x)
        x = self.mlp1(x)
        
        x = 0.8*torch.sigmoid(x)+0.1
        return x


class Classifier(nn.Module):
    """Given sub-image produces label"""

    def __init__(self, no_categories, no_in_channels=3):
        super().__init__()

        self.block1 = CNNBlock(3, 16)  # Bx3x16x16 -> Bx16x8x8
        self.block2 = CNNBlock(16, 16)  # Bx16x8x8   -> Bx16x4x4
        self.mlp = MLP([16*4*4, 64, 32, no_categories])

    def forward(self, x):
        """Classifies sub-image x"""

        x = self.block1(x)
        x = self.block2(x)
        x = self.mlp(x)
        x = F.gumbel_softmax(x, hard=True)
        return x


class NoPModel(nn.Module):
    """Network"""

    def __init__(self, no_categories, device):
        """Holds all models"""
        super(NoPModel, self).__init__()

        self.no_categories = no_categories
        self.preprocess = Preprocess()
        self.localiser = LocationModule(device)
        self.attention0 = Attention(no_categories)
        self.attention1 = Attention(no_categories)
        self.attention2 = UnlabelledAttention()
        self.classifier = Classifier(no_categories)
        self.device = device

    def parameters_all(self):
        params0 = list()
        for module in (self.preprocess, self.localiser, self.attention0, self.classifier):
            params0 += list(module.parameters())
        return params0, self.attention1.parameters(), self.attention2.parameters()

    def param0(self):
        params0 = list()
        for module in (self.preprocess, self.localiser, self.attention0, self.classifier):
            params0 += list(module.parameters())
        return params0

    def param1(self):
        return self.attention1.parameters()

    def param2(self):
        return self.attention2.parameters()

            
    def spatial(self, x, position):
        """Given position information return boundbox"""

        convert = [[[0.0,0,1,0],[0,0,0,0],[1,0,-0.5,0]],
                   [[0.0,0,0,0],[0,0,0,1],[0,1,0,-0.5]]]
        toAffine = torch.tensor(convert, device=self.device)
        theta=torch.einsum("xyz,bz->bxy", [toAffine,position])
        b, c, _, _ = x.shape
        newShape = torch.Size([b,c,16,16])
        grid = F.affine_grid(theta, newShape)
        x = F.grid_sample(x, grid)
        return x


    def forward(self, data):
        """Run an iteration"""
        randLabel = torch.rand([data.shape[0], self.no_categories], device=self.device)
        x, y = self.preprocess(data)
        a0 = self.attention0(y, randLabel)
        pos0 = self.localiser(x, a0)
        subx = self.spatial(data, pos0)
        labels = self.classifier(subx)
        targets = torch.argmax(labels, dim=1)
        a1 = self.attention1(y, labels)
        pos1 = self.localiser(x, a1)
        a2 = self.attention2(y)
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

def pos_str(pos):
    fs = "(x,y) = ({:#.2f},{:#.2f}), (sx,sy) = ({:#.2f},{:#.2f})"
    return fs.format(*pos)

class NoPNet():
    def __init__(self, args):
        """Trains the Naming of Parts network"""

        # define args
        self.args = args
        self.device = torch.device(self.args.device)
        if args.use_cuda:
            args.batch_size *= torch.cuda.device_count()

        # Define Networks
        self.model = NoPModel(self.args.no_categories, self.device)
        if self.args.use_cuda:
            self.model = model.to(self.device)
            #           if torch.cuda.device_count()>0:
            #               model = nn.DataParallel(model)

    def dataset(self, dataset_name):
        """Initialise Dataloaders"""
        # Dataset
        self.training_set =load_data(dataset_name, DataSetType.Train)
        self.testing_set = load_data(dataset_name, DataSetType.Test)

        self.trainloader = DataLoader(self.training_set,
                                      batch_size=self.args.batch_size,
                                      shuffle=True,
                                      num_workers=self.args.num_workers,
                                      pin_memory=self.args.use_cuda)
        
        self.testloader = DataLoader(self.testing_set,
                                     batch_size=self.args.batch_size, shuffle=False,
                                     num_workers=self.args.num_workers,
                                     pin_memory=self.args.use_cuda)
        



    def train(self, no_epochs: int)->dict:
        """Runs the training algorithm and returns resuls"""

        optimiser0 = optim.SGD(self.model.param0(), lr=self.args.learning_rate, momentum=0.99)
        optimiser1 = optim.SGD(self.model.param1(), lr=self.args.learning_rate, momentum=0.99)
        optimiser2 = optim.SGD(self.model.param2(), lr=self.args.learning_rate, momentum=0.99)


        Metrics = collections.namedtuple('Metrics', ['epoch', "kl1", "kl2"])
        results = []
        for epoch in range(no_epochs):
            for i, data in enumerate(self.trainloader):
                data = data.to(self.device)
                pos0, pos1, pos2, _ = self.model(data)
                loss1 = kl_loss(pos0, pos1)
                if i%3==0:
                    optimiser0.zero_grad();
                    loss1.backward()
                    optimiser0.step()
                else:
                    loss2 = kl_loss(pos0, pos2);
                    if i%3==1:
                        optimiser1.zero_grad();
                        loss2.backward(retain_graph=True)
                        optimiser1.step()
                    else:
                        loss = loss1-loss2
                        optimiser2.zero_grad();
                        loss.backward()
                        optimiser2.step()
                

            if epoch % 5 == 4:
                kl1 = Sos()
                kl2 = Sos()
                with torch.no_grad():
                    for i, data in enumerate(self.testloader):
                        data = data.to(self.device)
                        pos0, pos1, pos2, _ = self.model(data)
                        kl1 += kl_loss(pos0, pos1)
                        kl2 += kl_loss(pos0, pos2)
                    res = Metrics(epoch+1, str(kl1), str(kl2))
                    results.append(res._asdict())
                    print("{}, kl1 = {}, kl2 = {}".format(epoch+1, kl1, kl2))



        weight_filename = str(uuid.uuid4())
        self.args.weightFile = weight_filename
        torch.save(self.model, "../data/runs/weights/" + weight_filename)
        return results


    def examples(self, no_examples: int = 2):
        torch.no_grad()
        i = 0
        print("\n** Examples **")
        for data in (self.testing_set[i] for i in range(no_examples)):
            data = data.unsqueeze(0).to(self.device)
            output = self.model.forward(data)
            pos0, pos1, pos2, categ = (torch.flatten(x).tolist() for x in output)
            print("* Output for image ", i) 
            print("pos0 = " + pos_str(pos0))
            print("pos1 = " + pos_str(pos1))
            print("pos2 = " + pos_str(pos2))
            print("Category = {:d}".format(categ[0]))
