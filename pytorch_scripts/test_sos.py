# test_sos.py

def test_addTensor_count():
    from sos import Sos
    import torch

    sos = Sos()
    x = torch.rand(5,5,5)
    sos += x
    n = sos.number()
    sos += x
    assert n, sos.number() == (125, 250)
    
def test_addTensor():
    from sos import Sos
    import torch

    sos = Sos()
    x = torch.rand(5,5,5)
    sos += x
    print(repr(sos))
    print(str(sos))
    assert sos.av(), sos.var() == pytes.approx(torch.var(x.flatten()),
                                               torch.mean(x.flatten()))

def test_addItt():
    from sos import Sos
    import numpy as np

    x = np.random.rand(100)
    y = np.random.rand(100)

    sos = Sos()
    sos += x
    sos += y
    xy = np.concatenate([x,y])
    assert  sos.av(), sos.var() == pytest.approx(np.mean(xy), np.var(xy))
    
def test_str():
    from sos import Sos
    import numpy as np

    sos = Sos()
    sos += np.random.rand(100)
    print(repr(sos))
    print(sos)
    sos += np.random.rand(10000)
    print(repr(sos))
    print(sos)
 #   sos += np.random.rand(1000000)
 #   print(repr(sos))
 #   print(sos)
 #   print(sos.to_prec())
    assert True

def test_repr():
    from sos import Sos
    import numpy as np

    sos = Sos()
    sos += np.random.rand(100)
    print(repr(sos))
    assert True
