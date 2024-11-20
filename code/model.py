import torch.nn as nn
#imput size 28*28

# convolutional layer 1
# output = 28 - 5 + 1 = 24
# (1,28,28) -> (6, 24, 24)
conv_layer1 = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5,5)),
    nn.ReLU(),
)

# pooling layer 1 (2,2)
# output = 24/2 = 12
# (6, 24, 24) -> (6, 12, 12)

# convolutional layer 2
# output = 12 - 3 + 1 = 10
# (6, 12, 12) -> (32, 10, 10)
conv_layer2 = nn.Sequential(
    nn.Conv2d(in_channels=6, out_channels=32, kernel_size=(3,3)),
    nn.ReLU(),
)

#pooling layer 2 (2,2)
# output = 10/2 = 5
# (32, 10, 10) -> (32, 5, 5)

# convolutional layer 3
# output = 5 - 1 + 1 = 5
# (32, 5, 5) -> (16, 5, 5)
conv_layer3 = nn.Sequential(
    nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1,1)),
    nn.ReLU(),
)


# fully connected layer 1
fc_layer1 = nn.Sequential(
    nn.Linear(in_features=16*5*5, out_features=64),
    nn.ReLU(),
)

# fully connected layer 2
fc_layer2 = nn.Sequential(
    nn.Linear(in_features=64, out_features=10),
)

LeNet5 = nn.Sequential(
    conv_layer1,
    nn.MaxPool2d(kernel_size=(2,2)),
    conv_layer2,
    nn.MaxPool2d(kernel_size=(2,2)),
    conv_layer3,
    nn.Flatten(), # flatten
    fc_layer1,
    fc_layer2,
)

def count_parameters(layer):
    return sum(p.numel() for p in layer.parameters() if p.requires_grad)

print("Conv Layer 1 parameters:", count_parameters(conv_layer1))
print("Conv Layer 2 parameters:", count_parameters(conv_layer2))
print("Conv Layer 3 parameters:", count_parameters(conv_layer3))
print("FC Layer 1 parameters:", count_parameters(fc_layer1))
print("FC Layer 2 parameters:", count_parameters(fc_layer2))