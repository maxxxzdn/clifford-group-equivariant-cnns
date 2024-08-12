from .resnet import ResNet
from .csresnet import CSResNet
from .sresnet import SteerableResNet

try:
    from .cresnet import CliffordResNet
except ImportError:
    print("cliffordlayers not installed. Install it with 'pip install cliffordlayers'.")
