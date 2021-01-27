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
import numpy as np


###############################################################
#######  Blocks
###############################################################


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

###############################################################
#######  Preprocessing
###############################################################


class Preprocess(nn.Module):
    """Preprocess images using CNN"""
    
    def __init__(self, device):
        """Defines parameters for module"""
        super().__init__()
        self.device = device

        # CNN0
        self.block1 = CNNBlock(3,16)  # Bx3x64x64  -> Bx16x32x32
        self.block2 = CNNBlock(16,14) # Bx16x32x32 -> Bx16x16x16

        # CNN1
        self.block3 = CNNBlock(16, 16)   # Bx16x16x16 -> Bx16x8x8
        self.block4 = CNNBlock(16, 16)   # Bx16x8x8   -> Bx16x4x4
        self.mlp = MLP([16*4*4, 64, 32]) # Bx16x4x4   -> Bx32

    def forward(self, input: "Tensor") -> ("Tensor", "Tensor"):
        """Performs preprocessing 
        Input  x = [B, 3, 64, 64]
        Output x = [B, 16, 16, 16] block
        Output y = [B, 32]"""

        x = self.block1(input) # don't destroy input
        x = self.block2(x)

        # Add gradient vectors
        b, _, _, width = x.shape
        ones = torch.ones(width, device=self.device)
        seq = torch.linspace(0,1,width, device=self.device) 
        colCoord = torch.einsum("a,b->ab", [ones,seq]).repeat(b,1,1,1)
        rowCoord = torch.einsum("a,b->ab", [seq,ones]).repeat(b,1,1,1)
        x = torch.cat((x,colCoord,rowCoord), dim=1)

        # Attention block CNN1
        y = self.block3(x)
        y = self.block4(y)
        y = self.mlp(y)
        
        return x, y

###############################################################
#######  Attention Module
###############################################################


class Attention(nn.Module):
    """Given preproceesed image and label output attention mask"""

    def __init__(self, no_categories):
        """Holds parameters for module"""
        super().__init__()
        self.mlp0 = MLP([no_categories, 32])
        self.mlp1 = MLP([64, 32, 16])


    def forward(self, x0: "Tensor", label: "Tensor") -> "Tensor":
        """computes an attention mask
        Input    x0 = [B,32] # image attention gets re-used so don't destroy
        Input label = [B,no_categories]
        Output    x = [N,16] % channel normalisation"""

        label = self.mlp0(label)
        x = torch.cat([label, x0], dim=1)  # don't destroy x0
    
        x = self.mlp1(x)

        return x
    
class UnlabelledAttention(nn.Module):
    """Given preproceesed image and label output attention mask"""

    def __init__(self):
        """Holds parameters for module"""
        super().__init__()
        self.mlp1 = MLP([32, 32, 16])


    def forward(self, x0: "Tensor") -> "Tensor":
        """computes an attention mask
        Input     x0 = [B,32] : image attention gets reused so don't destroy
        Input  label = [B,no_categories]
        Output     x = [N,16] : channel normalisation"""

        x = self.mlp1(x0) # don't destroy x0 

        return x

###############################################################
#######  Localisation
###############################################################


class LocationModule(nn.Module):
    """Output Position"""

    def __init__(self, device):
        """Defines parameters for module"""
        super().__init__()
        self.block1 = CNNBlock(16, 16)  # Bx18x16x16 -> Bx16x8x8
        self.block2 = CNNBlock(16, 16)  # Bx16x8x8   -> Bx16x4x4
        self.mlp1 = MLP([16*4*4, 32, 4])
        self.device = device

        
    def forward(self, x0: "Tensor", att: "Tensor") -> "Tensor":
        """Locates an object in a image
        Input  x0 = [B, 16, 16, 16] gets re-used so don't destroy
        Input att = [B, 16]
        Output  x = [x, y, sx, sy]"""

        # multiplicative attention
        x = torch.einsum("bcwh,bc->bcwh", [x0,att]) # don't destroy x0

        # CNN + MLP
        x = self.block1(x)
        x = self.block2(x)
        x = self.mlp1(x)


        
        x = 0.8*torch.sigmoid(x)+0.1
        return x


###############################################################
#######  Classifier
###############################################################

class Classifier(nn.Module):
    """Given sub-image produces label"""

    def __init__(self, no_categories, no_in_channels=3):
        super().__init__()

        self.block1 = CNNBlock(3, 16)  # Bx3x16x16 -> Bx16x8x8
        self.block2 = CNNBlock(16, 16)  # Bx16x8x8   -> Bx16x4x4
        self.mlp = MLP([16*4*4, 64, 32, no_categories])

    def forward(self, x):
        """Classifies sub-image x
           Don't need to keep input"""

        x = self.block1(x)
        x = self.block2(x)
        x = self.mlp(x)
        return x

###############################################################
#######  Model
###############################################################

class NoPModel(nn.Module):
    """Network"""

    def __init__(self, args, device):
        """Holds all models"""
        super(NoPModel, self).__init__()

        self.no_categories = args["no_categories"]
        self.batch_size = args["batch_size"]
        self.replicas = args["replicas"]
        self.preprocess = Preprocess(device)
        self.localiser = LocationModule(device)
        self.attention0 = Attention(self.no_categories)
        self.attention1 = Attention(self.no_categories)
        self.classifier = Classifier(self.no_categories)
        self.device = device

            
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


    def forward(self, originalImages):
        """Run an iteration"""

        # Preprocess
        x, y = self.preprocess(originalImages)

        B = x.shape[0]

        # create a set of identical images
        x = torch.cat(self.replicas*[x])
        y = torch.cat(self.replicas*[y])
        originalImages = torch.cat(self.replicas*[originalImages])

        # Use attention with random labels
        randLabel = torch.rand([B*self.replicas,
                                self.no_categories], device=self.device)
        a0 = self.attention0(y, randLabel)
        pos0 = self.localiser(x, a0)
        del a0

        # Use spatial transofrm to cut out subimage and classify
        subx = self.spatial(originalImages, pos0)
        labels0 = self.classifier(subx)
        del subx

        # Attention based on hard lables
        hardLabels = F.gumbel_softmax(labels0, hard=True)
        a1 = self.attention1(y, hardLabels)
        del hardLabels
        pos1 = self.localiser(x, a1)
        del a1

        # Classify cut out image
        subx = self.spatial(originalImages, pos0)
        labels1 = self.classifier(subx)
        
        return pos0, pos1, labels0, labels1

###############################################################
#######  Helper Functions and Classes
###############################################################

        
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

class LossMetric():
    
    def __init__(self, name, epoch):
        self.data = {"name":name, "epoch":epoch,
                     "diversity":Sos(), "kl":Sos(), "labels":Sos()}
    
    def to_dict(self):
        return {"epoch": self.data["epoch"],
                "diversity":str(self.data["diversity"]),
                "kl":str(self.data["kl"]),
                "labels":str(self.data["labels"]),}
    
    def __str__(self):
        fs = "{}: {}, kl = {}, diversity = {}, labels = {}"
        return fs.format(self.data["name"],
                         self.data["epoch"],
                         self.data["kl"],
                         self.data["diversity"],
                         self.data["labels"])

###############################################################
#######  Driver
###############################################################


class NoPNet():
    def __init__(self, args):
        """Trains the Naming of Parts network"""

        # define args
        self.args = args
        self.device = torch.device(self.args["device"])

        # Define Networks
        self.model = NoPModel(self.args, self.device)
        if self.args["use_cuda"]:
#            args["batch_size"] *= torch.cuda.device_count()
            self.model = self.model.to(self.device)
            #           if torch.cuda.device_count()>0:
            #               model = nn.DataParallel(model)

    def dataset(self, dataset_name, record):
        """Initialise Dataloaders"""
        # Dataset
        self.training_set =load_data(dataset_name, DataSetType.Train)
        self.testing_set = load_data(dataset_name, DataSetType.Test)

        self.trainloader = DataLoader(self.training_set,
                                      batch_size=self.args["batch_size"],
                                      shuffle=True,
                                      drop_last = True,
                                      num_workers=self.args["num_workers"],
                                      pin_memory=self.args["use_cuda"])
        
        self.testloader = DataLoader(self.testing_set,
                                     batch_size=self.args["batch_size"],
                                     shuffle=False,
                                     drop_last = True,
                                     num_workers=self.args["num_workers"],
                                     pin_memory=self.args["use_cuda"])
        
        record["dataset"] =  {"name": dataset_name,
                              "imageSize": self.training_set[0].shape,
                              "training_size": len(self.training_set),
                              "test_size": len(self.testing_set)}
        

    def loss(self, data, metric):
        B, R, C = (self.args["batch_size"], self.args["replicas"],
                   self.args["no_categories"])
        data = data.to(self.device)
        pos0, pos1, labels0, labels1 = self.model(data)
        softlabels = F.softmax(labels0)
        ls = F.log_softmax(labels0)
        lossLabel = - torch.einsum("bl,bl->", [softlabels, ls])
        softlabels = softlabels.view(B,R,C)
        sumLabels = torch.sum(softlabels, dim=1)
        sumLabels = 2*torch.sigmoid(2.0*sumLabels)-1
        lossDiversity = R - torch.mean(sumLabels)
        lossKL = kl_loss(pos0, pos1)
        #        with torch.no_grad():
        metric.data["diversity"] += float(lossDiversity)
        metric.data["kl"] += float(lossKL)
        metric.data["labels"] += float(lossLabel.item())
        return 5*lossDiversity + lossKL + lossLabel
    

    def train(self, no_epochs: int, record: dict):
        """Runs the training algorithm and returns resuls"""

        optimiser = optim.SGD(self.model.parameters(),
                              lr=self.args["learning_rate"],
                              momentum=0.99)


        
        resTrain = []
        resTest = []
        for epoch in range(no_epochs):
            diversity = Sos()
            kl = Sos()
            trainingLoss = LossMetric("Train", epoch)
            for i, data in enumerate(self.trainloader):
                optimiser.zero_grad();
                loss = self.loss(data, trainingLoss)
                loss.backward()
                optimiser.step()
                if (i%100==99):
                    print(i, trainingLoss)
                
            print(trainingLoss)
            resTrain.append(trainingLoss.to_dict())
            
            if True:
                testingLoss = LossMetric("Test", epoch)
                with torch.no_grad():
                    for i, data in enumerate(self.testloader):
                        loss = self.loss(data, testingLoss)
                    resTest.append(testingLoss.to_dict())
                    print(testingLoss)

        weight_filename = str(uuid.uuid4())
        self.args['weightFile'] = weight_filename
        torch.save(self.model.state_dict(), "../data/runs/weights/" + weight_filename)

        print(resTrain)
        print(resTest)
        record["training_data"] = {"train": resTrain, "test": resTest}


    def load(self):
        weight_filename = "../data/runs/weights/" + self.args['weightFile']
        state_dict = torch.load(weight_filename)
        self.model.load_state_dict(state_dict)

    def examples(self, record, no_examples: int = 2):
        torch.no_grad()
        i = 0
        print("\n** Examples **")
        for data in (self.testing_set[i] for i in range(no_examples)):
            data = data.unsqueeze(0).to(self.device)
            output = self.model.forward(data)
            pos0, pos1, one_hot = output
            pos0 = pos0.tolist()
            pos1 = pos1.tolist()
            labels = torch.argmax(one_hot, dim=1)
            print(labels)
            print(pos0)
            for p0, p1, label in zip(pos0, pos1, labels):
                print("* Output for image ", i) 
                print("pos0 = " + pos_str(p0))
                print("pos1 = " + pos_str(p1))
                print("Category = ", label.item())

    def printParameters(self):
        total_param = 0
        for name, param in self.model.named_parameters():
            noParam = param.storage().size()
            total_param += noParam
            print(name, param.shape, noParam)
        print("Total number of parameters", total_param)
