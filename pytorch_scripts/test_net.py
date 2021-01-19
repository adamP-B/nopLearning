# test_net.py

import math
import pytest
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

from mydataset import load_data, DataSetType
from NoPNet import Preprocess, Attention, UnlabelledAttention
from NoPNet import LocationModule, Classifier, NoPModel
from NoPNet import kl_loss

batch_size = 10
no_categories = 22
device = torch.device('cpu')

# Test dataloader

@pytest.fixture
def batch():
    clevr = load_data("CLEVR", DataSetType.Train)
    dataloader = DataLoader(clevr, batch_size=batch_size, shuffle=True)
    return dataloader.__iter__().next()

def test_batch(batch):
    assert batch.shape == torch.Size([batch_size, 3, 64, 64])


# Test  Preprocee

@pytest.fixture
def preprocessor():
    return Preprocess()

def test_preprocess_init(preprocessor):
    assert issubclass(type(preprocessor), nn.Module)

    
def test_preprocess_x(batch, preprocessor):
    x, _ = preprocessor(batch)
    assert x.shape == torch.Size([batch_size,16,16,16])

def test_preprocess_label(batch, preprocessor):
    _, y = preprocessor(batch)
    assert y.shape == torch.Size([batch_size,32])

# Test Attention

@pytest.fixture
def attention():
    return Attention(no_categories)

@pytest.fixture
def attentionNone():
    return UnlabelledAttention()

@pytest.fixture
def attention_data():
    return torch.ones([batch_size,32]), torch.ones([batch_size, no_categories])

def test_attention_data(attention_data):
    assert attention_data[0].shape == torch.Size([batch_size,32])
    assert attention_data[1].shape == torch.Size([batch_size, no_categories])

def test_attention_init(attention):
    assert issubclass(type(attention), nn.Module)

    
def test_attention(attention_data, attention):
    x = attention(*attention_data)
    assert x.shape == torch.Size([batch_size,16])

def test_attentionNone(attention_data, attentionNone):
    x = attentionNone(attention_data[0])
    assert x.shape == torch.Size([batch_size,16])

# Test Location Module


@pytest.fixture
def location():
    return LocationModule(device)


@pytest.fixture
def location_data():
    return torch.ones([batch_size,16,16,16]), torch.ones([batch_size,16])

def test_location_data(location_data):
    assert location_data[0].shape == torch.Size([batch_size,16,16,16])
    assert location_data[1].shape == torch.Size([batch_size,16])

def test_location_init(location):
    assert issubclass(type(location), nn.Module)

    
def test_location(location_data, location):
    x = location(*location_data)
    assert x.shape == torch.Size([batch_size,4])



# Test Location Module


@pytest.fixture
def classifier():
    return Classifier(no_categories)


@pytest.fixture
def classifier_data():
    return torch.ones([batch_size,3,16,16])

def test_classifier_data(classifier_data):
    assert classifier_data.shape == torch.Size([batch_size,3,16,16])

def test_classifier_init(classifier):
    assert issubclass(type(classifier), nn.Module)

    
def test_classifier(classifier_data, classifier):
    x = classifier(classifier_data)
    assert x.shape == torch.Size([batch_size,no_categories])


# Test NoPModel

@pytest.fixture
def model():
    return NoPModel(no_categories, device)

def test_model_init(model):
    assert issubclass(type(model), nn.Module)

    
def test_model(batch, model):
    x = model(batch)
    x_shapes = [t.shape for t in x[0:3]]
    ref_shapes = [torch.Size([batch_size,4]) for i in range(3)]
    assert x_shapes == ref_shapes

def test_model_target(batch, model):
    x = model(batch)
    assert x[3].shape == torch.Size([batch_size])

def test_model_target_out(batch, model):
    x = model(batch)
    assert all([(c<no_categories) & (c>=0) for c in x[3]])

    
def test_model(batch, model):
    x = model(batch)
    x_shapes = [t.shape for t in x[0:3]]
    ref_shapes = [torch.Size([batch_size,4]) for i in range(3)]
    assert x_shapes == ref_shapes

def test_model_param(model):
#    for p in model.parameters_all():
#        print(p)
    assert len(model.parameters_all()) == 3
    
# Test KLdivergence

def test_kl():
    p1 = torch.tensor([0.0,0,1,1])
    p2 = torch.tensor([0.1,-0.2,0.5,0.5])
    pos1 = torch.stack((p1, p2))
    pos2 = torch.stack((p1, p2))
    kl = kl_loss(pos1, pos2)
    print(kl)
    assert kl==0.00
 
def test_kl1():
    mu1 = 0.0
    mu2 = 0.1
    s1 = 1.0
    s2 = 0.5
    p1 = torch.tensor([mu1,0,s1,1])
    p2 = torch.tensor([mu2,0,s2,1])
    pos1 = torch.stack((p1, p1))
    pos2 = torch.stack((p1, p2))
    kl = kl_loss(pos1, pos2)
    r = (s1/s2)**2
    mykl = (r - 1 - math.log(r) + (mu1-mu2)**2/s2**2)/2
    print(kl, mykl)
    assert kl==mykl
