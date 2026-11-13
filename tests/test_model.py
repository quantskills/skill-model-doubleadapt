import torch
from doubleadapt.model import DoubleAdaptModel
def test_forward_shapes():
    model=DoubleAdaptModel(5,hidden_size=8,layers=2,heads=2)
    x=torch.randn(7,5)
    assert model(x).shape==(7,)
    assert model.adapted_label(x,torch.randn(7)).shape==(7,)
