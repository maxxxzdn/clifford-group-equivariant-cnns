from .resnet import ResNet
from .csresnet import CSResNet

try:
    from .sresnet import SteerableResNet
except ImportError:
    print("escnn not installed. Install it with 'pip install escnn'.")

try:
    from .cresnet import CliffordResNet
except ImportError:
    print("cliffordlayers not installed. Install it with 'pip install cliffordlayers'.")
