from .resnet_cifar import resnet18 as resnet18_cifar
MODELS = {
    'cifar10': {
        'resnet18': resnet18_cifar,
    },
}