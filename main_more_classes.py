import numpy as np
from comet_ml import Experiment
import torch
import torch.nn.functional as F
import torchvision
from matplotlib import pyplot as plt
from torch import nn
from torchvision.transforms import transforms, RandAugment

PATH = '/home/aherkov/PycharmProjects/seminar2/models/model_aug_no_outliers_5_classes.pth'
experiment = Experiment(api_key="OxtabtUxzHFJ58bIaa8tZ69Ll", project_name="model_aug_no_outliers_5_classes")

torch.cuda.empty_cache()


def get_data_loader(data_dir, train=False):
    if train:
        tea_dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=transforms.Compose([
            RandAugment(),
            transforms.Resize([96, 96]),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ]))
    else:
        tea_dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=transforms.Compose([
            transforms.Resize([96, 96]),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ]))

    data_loader = torch.utils.data.DataLoader(dataset=tea_dataset, batch_size=4, shuffle=True, num_workers=4)
    return data_loader


class SmallNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, (3, 3), (1, 1), bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d((2, 2), (2, 2)),
            nn.Conv2d(16, 32, (3, 3), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(32, 32, (3, 3), (1, 1), bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d((2, 2), (2, 2)),
            nn.Flatten(1),
            nn.Linear(14112, 1024),
            nn.Dropout(0.25),
            nn.Linear(1024, 5)
        )

    def forward(self, x):
        x = self.network(x)
        output = F.log_softmax(x, dim=1)
        return output


small_net = SmallNet()

if torch.cuda.is_available():
    small_net = small_net.cuda()

train_data_loader = get_data_loader("/home/aherkov/PycharmProjects/seminar2/images/5classes_no_outliers/train", train=True)
test_data_loader = get_data_loader("/home/aherkov/PycharmProjects/seminar2/images/5classes_no_outliers/test")
validation_data_loader = get_data_loader("/home/aherkov/PycharmProjects/seminar2/images/5classes_no_outliers/validation")

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(small_net.parameters(), lr=0.0001)

prev_acc = 0
trigger = 0
patience = 2


def train(epoch):
    running_loss = 0.0
    for i, data in enumerate(train_data_loader):
        inputs, labels = data
        if torch.cuda.is_available():
            inputs, labels = inputs.cuda(), labels.cuda()

        optimizer.zero_grad()
        outputs = small_net(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        experiment.log_metric("loss", loss.item(), step=i)

        running_loss += loss.item()
        if i % 1000 == 999:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 1000:.3f}')
            running_loss = 0.0


def validate(epoch):
    global trigger, prev_acc
    correct = 0
    total = 0
    with torch.no_grad():
        for data in validation_data_loader:
            images, labels = data
            images, labels = images.cuda(), labels.cuda()
            outputs = small_net(images)
            _, predictions = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    curr_acc = 100 * correct / total
    experiment.log_metric("validation accuracy", curr_acc, step=epoch)
    print("current accuracy:", curr_acc)

    if curr_acc < prev_acc:
        trigger += 1
    else:
        trigger = 0
    prev_acc = curr_acc


with experiment.train():
    for epoch in range(0, 40):
        train(epoch)
        validate(epoch)
        if trigger >= patience:
            print('validation accuracy not improving. early stopping.')
            break

torch.save(small_net.state_dict(), PATH)
classes = ('AppleCinnamonTeaBox', 'AroniaTeaBox', 'BlueberryTeaBox', 'CamomileTeaBox', 'GreenTeaBox')

dataiter = iter(test_data_loader)
images, labels = dataiter.next()


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

net = SmallNet()
net.load_state_dict(torch.load(PATH))
outputs = net(images)
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(4)))


def test():
    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_data_loader:
            images, labels = data
            images, labels = images.cuda(), labels.cuda()
            outputs = small_net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'classification accuracy on test data: {100 * correct // total}%')
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    with torch.no_grad():
        for data in test_data_loader:
            images, labels = data
            images, labels = images.cuda(), labels.cuda()
            outputs = small_net(images)
            _, predictions = torch.max(outputs, 1)
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'accuracy for class: {classname:5s} is {accuracy:.1f} %')


test()
del small_net
