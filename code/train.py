import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
drawfigure = True
if drawfigure == True:
    import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import load_mnist, load_cifar10, load_fashion_mnist, imshow
from model import LeNet5

# test
@torch.no_grad()
def accuracy(model, data_loader):
    model.eval()
    correct, total = 0, 0
    for batch in data_loader:
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return correct / total

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

trainset, testset = load_mnist()


#Implement a preprocessing step 
#by applying torchvision.transforms.Grayscale 
# and torchvision.transforms.
# Resize to ensure the input image is the correct size. (1 point)
transform = transforms.Compose(
    [transforms.Grayscale(),
     transforms.Resize((28, 28))])

trainloader = DataLoader(trainset, batch_size=8, shuffle=True)
testloader = DataLoader(testset, batch_size=8, shuffle=False)


model = LeNet5.to(device)

loss_fn = nn.CrossEntropyLoss()
#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 4], gamma=0.1)

# loop over the dataset multiple times
if drawfigure == True:
    table_name = "Adam"
    loss_list = []
    accuracy_list = []
num_epoch = 5
model.train()
for epoch in range(num_epoch):
    running_loss = 0.0
    for i, batch in enumerate(trainloader, 0):
        # get the images; batch is a list of [images, labels]
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()  # zero the parameter gradients
        images = transform(images)
        # get prediction
        outputs = model(images)
        # compute loss
        loss = loss_fn(outputs, labels)
        # reduce loss
        loss.backward()
        optimizer.step()


        # print statistics
        running_loss += loss.item()
        if i % 1000 == 999:  # print every 500 mini-batches
            if drawfigure == True:
                loss_list.append(running_loss / 1000,)
                accuracy_list.append(accuracy(model, testloader))
            print('[epoch %2d, batch %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 1000))
            running_loss = 0.0
            # train_acc =
            # test_acc =
            # print('Accuracy on the train set: %f %%' % (100 * train_acc))
            # print('Accuracy on the test set: %f %%' % (100 * test_acc))
            # print('The current learning rate is: %f ' % (scheduler.get_last_lr()[0]))
    scheduler.step()


if drawfigure == True:
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




model_file = 'model.pth'
torch.save(model.state_dict(), model_file)
print(f'Model saved to {model_file}.')

print('Finished Training')


# show some prediction result
dataiter = iter(testloader)
# images, labels = dataiter.next()
images, labels = next(dataiter)
images = images.to(device)
predictions = model(images).argmax(1).detach().cpu()

classes = trainset.classes
print('GroundTruth: ', ' '.join('%5s' % classes[i] for i in labels))
print('Prediction: ', ' '.join('%5s' % classes[i] for i in predictions))
imshow(torchvision.utils.make_grid(images.cpu()))




# train_acc = accuracy(model, trainloader)
test_acc = accuracy(model, testloader)

# print('Accuracy on the train set: %f %%' % (100 * train_acc))
print('Accuracy on the test set: %f %%' % (100 * test_acc))
