# test_dataset.py

import pytest
import torch

from mydataset import load_data, DataSetType
from torch.utils.data import Dataset


@pytest.fixture
def clevr():
    return load_data("CLEVR", DataSetType.Train)

def test_load_data(clevr):
    """Test load data"""
    assert issubclass(type(clevr),Dataset)
        
def test_dataset_iterator(clevr):
    """Test load data"""
    assert issubclass(type(clevr[0]), torch.Tensor)

def test_dataset_data(clevr):
    assert clevr[-1].shape == torch.Size([3,64,64])
