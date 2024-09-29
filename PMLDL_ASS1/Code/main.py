import torch
import tensorflow as tf
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    print("[*] Loading data...")

    image_size = 256

    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])

    test_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])

    dataset_path = 'D:\PMLDL_ASS1\PMLDL_ASS1\Code\Datasets'

    train_dataset = datasets.ImageFolder(root=dataset_path, transform=train_transform)
    test_dataset = datasets.ImageFolder(root=dataset_path, transform=test_transform)

    print("[*] Data loaded.")
    print("[*] Splitting dataset...")

    val_ratio = 0.1
    random_state = 200
    indices = list(range(len(train_dataset)))
    train_indices, val_indices = train_test_split(indices, test_size=val_ratio, random_state=random_state)

    train_dataloader = DataLoader(train_indices, batch_size=128, shuffle=True)
    val_dataloader = DataLoader(val_indices, batch_size=128, shuffle=True)

    train_original = [test_dataset[index][0] for index in train_indices]
    val_original = [test_dataset[index][0] for index in val_indices]

    print("[*] Data split.")

    plot_image_flag = True

    if (plot_image_flag):
        print("[*] Plotting images...")
        for batch in val_dataloader:
            image_indexes = batch

            rows = 3
            fig = plt.figure(figsize=(10, 7))

            for i in range(rows):

                train_image = train_dataset[image_indexes[i]][0]
                test_image = test_dataset[image_indexes[i]][0]

                fig.add_subplot(rows, 2, i * 2 + 1)
                plt.tight_layout()
                plt.axis('off')
                plt.title('Grayscale')
                plt.imshow(train_image.numpy().transpose((1, 2, 0)), cmap='gray')

                fig.add_subplot(rows, 2, i * 2 + 2)
                plt.tight_layout()
                plt.axis('off')
                plt.title('Original')
                plt.imshow(test_image.numpy().transpose((1, 2, 0)), cmap='gray')

            plt.show()
        print("[*] Images plotted.")

