#my model was trained on the fashion_mnist dataset
# in the first 10 epochs, I used SGD with a learning rate of 0.0001 and a momentum of 0.9
# the accuracy was reached 0.90
# in the next 20 epochs, I used Adam with a learning rate of 0.0001
# the accuracy was reached 0.93, which is what i can achieve in my laptop with nvidia 4060 8G VRAM
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
drawfigure = True
if drawfigure == True:
    import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from dataset import load_fashion_mnist

if torch.cuda.is_available():
    device = torch.device(0)
else:
    device = torch.device('cpu')

def draw_loss_accuracy(loss_list, accuracy_list, table_name):
    plt.plot(loss_list, label='loss')
    plt.plot(accuracy_list, label='accuracy')
    plt.xlabel('batch')
    plt.ylabel('loss')
    plt.ylim(0, 1.2)
    plt.title(table_name+' Training loss and accuracy')
    plt.legend()
    for i, acc in enumerate(accuracy_list):
        plt.annotate(
            f'{acc*100:.2f}%', 
            (i, acc), 
            textcoords="offset points", 
            xytext=(0,10), 
            ha='center',
            rotation=45
            )
    for i, loss in enumerate(loss_list):
        plt.annotate(
            f'{loss:.2f}', 
            (i, loss), 
            textcoords="offset points", 
            xytext=(0,10), 
            ha='center',
            rotation=45
            )
    plt.savefig(table_name + '_loss.png')
    plt.close()

## test
@torch.no_grad()
def accuracy(model, data_loader):
    model.eval()
    correct, total = 0, 0
    for batch in data_loader:
        images, labels = batch
        processed_images = []
        for image in images:
            #resize the image to 224*224
            image_np = image.numpy()
            image_np = image_np.squeeze()
            image_np = image_np * 255
            image = Image.fromarray(image_np)
            #save the first image

            image = image.convert('L')
            image = preprocess(image)
            processed_images.append(image)

        images = torch.stack(processed_images)
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return correct / total


trainset, testset = load_fashion_mnist()
# Get the classes
classes = trainset.classes
print("Classes:", classes)

# Alternatively, you can get the unique labels from the targets
unique_labels = set(trainset.targets.numpy())
print("Unique labels:", unique_labels)


trainloader = DataLoader(trainset, batch_size=256, shuffle=True)
testloader = DataLoader(testset, batch_size=256, shuffle=False)


## your code here
# TODO: load ResNet18 from PyTorch Hub, and train it to achieve 90+% classification accuracy on Fashion-MNIST.
model = torch.hub.load('pytorch/vision:v0.10.0',
                       'resnet18',
                        weights="ResNet18_Weights.DEFAULT")

model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model.fc = nn.Linear(model.fc.in_features, 10)
import os
if os.path.exists("fashion_mnist.pth"):
    model.load_state_dict(torch.load("fashion_mnist.pth"))
model = model.to(device)
preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
])
preprocess_train = transforms.Compose([
    transforms.Resize(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomRotation(90),
    transforms.ToTensor(),
])

# print("acciracy: "+str(accuracy(model, testloader)))


loss_fn = nn.CrossEntropyLoss()
#optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
if drawfigure == True:
    table_name = "Adam_Rotation"
    loss_list = []
    accuracy_list = []
# loop over the dataset multiple times
num_epoch = 100
model.train()
for epoch in range(num_epoch):
    running_loss = 0.0
    for i, batch in enumerate(trainloader, 0):
        optimizer.zero_grad()  # zero the parameter gradients

        # get the images; batch is a list of [images, labels]
        images, labels = batch

        #save the first image
        # torchvision.utils.save_image(images[0], "pic/fashion_mnist_before_process.png")
        processed_images = []
        for image in images:
            #resize the image to 224*224
            image_np = image.numpy()
            image_np = image_np.squeeze()
            image_np = image_np * 255
            image = Image.fromarray(image_np)
            #save the first image

            image = image.convert('L')
            # image.save("pic/fashion_mnist_first_process.png")
            image = preprocess_train(image)
            # torchvision.utils.save_image(image, "pic/${i}fashion_mnist.png")
            processed_images.append(image)

        images = torch.stack(processed_images)
        #save the first image
        torchvision.utils.save_image(images[0], "pic/${i}fashion_mnist.png")
        # get prediction
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # compute loss
        loss = loss_fn(outputs, labels)
        # reduce loss
        loss.backward()
        optimizer.step()


        # print statistics
        running_loss += loss.item()
        if i % 10 == 9:  # print every 500 mini-batches
            print('[epoch %2d, batch %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 10))
            # train_acc =
            # print('The current learning rate is: %f ' % (scheduler.get_last_lr()[0]))
            if i%50 == 49:
                test_acc = accuracy(model, testloader)
                if drawfigure == True:
                    loss_list.append(running_loss / 10,)
                    accuracy_list.append(test_acc)
                    draw_loss_accuracy(loss_list, accuracy_list, table_name+'_'+str(i))
                # print('Accuracy on the train set: %f %%' % (100 * train_acc))
                print('Accuracy on the test set: %f %%' % (100 * test_acc))
                torch.save(model.state_dict(), "fashion_mnist.pth")
                if  test_acc > 0.95:
                    break
            running_loss = 0.0






train_acc = accuracy(model, trainloader)
test_acc = accuracy(model, testloader)

print('Accuracy on the train set: %f %%' % (100 * train_acc))
print('Accuracy on the test set: %f %%' % (100 * test_acc))
