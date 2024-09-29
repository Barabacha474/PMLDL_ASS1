import torch
import tensorflow as tf
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])

    dataset_path = 'D:\PMLDL_ASS1\PMLDL_ASS1\Code\Datasets'

    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    plot_image_flag = True

    if (plot_image_flag):
        for batch in dataloader:
            images, labels = batch
            plt.imshow(images[0].numpy().transpose((1, 2, 0)))
            plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
