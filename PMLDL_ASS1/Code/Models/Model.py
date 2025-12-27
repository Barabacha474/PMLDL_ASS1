import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.dropout = nn.Dropout(0.2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.conv1_2 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv2 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv2_2 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv3 = nn.Conv2d(512, 1024, 3, 1, 1)
        self.fc1 = nn.Linear(1024 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 100)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv1_2(x))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv2_2(x))
        x = self.dropout(x)
        x = F.relu(self.conv3(x))
        x = self.dropout(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

    def Get_class_from_prediction(self, index):
        return classes.get(index)

classes = {0: 'apple',
        1: 'aquarium_fish',
        2: 'baby',
        3: 'bear',
        4: 'beaver',
        5: 'bed',
        6: 'bee',
        7: 'beetle',
        8: 'bicycle',
        9: 'bottle',
        10: 'bowl',
        11: 'boy',
        12: 'bridge',
        13: 'bus',
        14: 'butterfly',
        15: 'camel',
        16: 'can',
        17: 'castle',
        18: 'caterpillar',
        19: 'cattle',
        20: 'chair',
        21: 'chimpanzee',
        22: 'clock',
        23: 'cloud',
        24: 'cockroach',
        25: 'couch',
        26: 'crab',
        27: 'crocodile',
        28: 'cup',
        29: 'dinosaur',
        30: 'dolphin',
        31: 'elephant',
        32: 'flatfish',
        33: 'forest',
        34: 'fox',
        35: 'girl',
        36: 'hamster',
        37: 'house',
        38: 'kangaroo',
        39: 'computer_keyboard',
        40: 'lamp',
        41: 'lawn_mower',
        42: 'leopard',
        43: 'lion',
        44: 'lizard',
        45: 'lobster',
        46: 'man',
        47: 'maple_tree',
        48: 'motorcycle',
        49: 'mountain',
        50: 'mouse',
        51: 'mushroom',
        52: 'oak_tree',
        53: 'orange',
        54: 'orchid',
        55: 'otter',
        56: 'palm_tree',
        57: 'pear',
        58: 'pickup_truck',
        59: 'pine_tree',
        60: 'plain',
        61: 'plate',
        62: 'poppy',
        63: 'porcupine',
        64: 'possum',
        65: 'rabbit',
        66: 'raccoon',
        67: 'ray',
        68: 'road',
        69: 'rocket',
        70: 'rose',
        71: 'sea',
        72: 'seal',
        73: 'shark',
        74: 'shrew',
        75: 'skunk',
        76: 'skyscraper',
        77: 'snail',
        78: 'snake',
        79: 'spider',
        80: 'squirrel',
        81: 'streetcar',
        82: 'sunflower',
        83: 'sweet_pepper',
        84: 'table',
        85: 'tank',
        86: 'telephone',
        87: 'television',
        88: 'tiger',
        89: 'tractor',
        90: 'train',
        91: 'trout',
        92: 'tulip',
        93: 'turtle',
        94: 'wardrobe',
        95: 'whale',
        96: 'willow_tree',
        97: 'wolf',
        98: 'woman',
        99: 'worm'}

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    def imshow(img):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 32

    trainset = torchvision.datasets.CIFAR100(root='./Data', train=True,
                                            download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR100(root='./Data', train=False,
                                           download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    net = Net()
    net.to(device)

    import torch.optim as optim

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    epochs = 15

    for epoch in range(epochs):  # loop over the dataset multiple times

        train_loop = tqdm(enumerate(trainloader), total=len(trainloader), desc=f"Epoch {epoch}")

        running_loss = 0.0
        for i, data in train_loop:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # print statistics
            train_loop.set_postfix({"Validation loss": running_loss / (i + 1)})

    print('Finished Training')

    try:
        PATH = 'PMLDL_ASS1\Models\Best_model.pt'
        torch.save(net.state_dict(), PATH)
    except:
        PATH = 'D:\PMLDL_ASS1\PMLDL_ASS1\Models\Best_model.pt'
        torch.save(net.state_dict(), PATH)

    dataiter = iter(testloader)
    images, labels = next(dataiter)

    images = images.to(device)
    labels = labels.to(device)

    # print images
    imshow(torchvision.utils.make_grid(images.cpu()))
    ground_truth_str = ''
    for i in range(batch_size):
        index = int(labels[i].cpu().numpy())
        ground_truth_str += classes.get(index) + ' '

    print('GroundTruth: ', ground_truth_str)

    net = Net()
    net.load_state_dict(torch.load(PATH, weights_only=True))
    net.to(device)

    outputs = net(images)

    _, predicted = torch.max(outputs, 1)

    predicted_str = ''
    for i in range(batch_size):
        index = int(predicted[i].cpu().numpy())
        predicted_str += classes.get(index) + ' '
    print('Predicted: ', predicted_str)

    val_loss = 0.0
    net.eval()
    val_loop = tqdm(enumerate(testloader), total=len(testloader), desc=f"Epoch {epoch}")

    for i, data in val_loop:
        inputs, labels = data[0].to(device), data[1].to(device)

        outputs = net(inputs)
        loss = criterion(outputs, labels)

        val_loss += loss.item()

        # print statistics
        val_loop.set_postfix({"Validation loss": val_loss / (i + 1)})
