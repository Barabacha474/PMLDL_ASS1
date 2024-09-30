import torch
import torch.nn as nn
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class ColorizationNet(nn.Module):
    def __init__(self, img_size, dropout_rate=0.5):
        super(ColorizationNet, self).__init__()

        self.dropout_rate = dropout_rate

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), #256
            nn.ReLU(),
            nn.MaxPool2d(2), # 128
            nn.Dropout(p=self.dropout_rate),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 64
            nn.Dropout(p=self.dropout_rate),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), #32
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 32 * 32 + 1 * img_size * img_size, 2056),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(2056, 3 * img_size * img_size),
        )

    def forward(self, x):
        x_conv = self.conv_layers(x)

        # print(x_conv.shape)
        # print(x.shape)

        x_conv = torch.flatten(x_conv, 0)
        x = torch.flatten(x, 0)

        # print(x_conv.shape)
        # print(x.shape)

        x = torch.cat([x_conv, x], dim=0)

        x = self.fc_layers(x)

        x = torch.sigmoid(x)

        x = x.reshape(3, 256, 256)

        # print(x.shape)

        return x

if __name__ == '__main__':
    print("[*] Loading data...")

    image_size = 256

    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])

    test_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])

    dataset_path = 'Code\Datasets'

    train_dataset = datasets.ImageFolder(root=dataset_path, transform=train_transform)
    test_dataset = datasets.ImageFolder(root=dataset_path, transform=test_transform)

    print("[*] Data loaded.")
    print("[*] Splitting dataset...")

    val_ratio = 0.1
    random_state = 200
    indices = list(range(len(train_dataset)))
    train_indices, val_indices = train_test_split(indices, test_size=val_ratio, random_state=random_state)

    batch_size = 34

    train_dataloader = DataLoader(train_indices, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_indices, batch_size=batch_size, shuffle=True)

    print("[*] Data split.")

    plot_load_image_flag = False
    plot_train_image_flag = True
    plot_val_image_flag = True

    if plot_load_image_flag:
        print("[*] Plotting images...")
        for batch in val_dataloader:
            image_indexes = batch

            rows = 3
            fig = plt.figure(figsize=(2 * 10, rows * 10))

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

    print("[*] Creating model...")

    model = ColorizationNet(image_size)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    num_epochs = 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"[!] Devise used: {device}")

    def train(
            model,
            optimizer,
            loss_fn,
            train_loader,
            val_loader,
            writer=None,
            epochs=1,
            device=device,
            ckpt_path='Models/best.pt',
    ):
        model.to(device)

        best = 0.0

        for epoch in range(epochs):

            train_loop = tqdm(
                enumerate(train_loader, 0), total=len(train_loader), desc=f"Epoch {epoch}"
            )
            model.train()
            train_loss = 0.0

            for k, (img_indexes) in train_loop:

                optimizer.zero_grad()

                outputs = []
                test_images = []

                plot_counter = 0
                max_plot_counter = 1

                fig_train = plt.figure(figsize=(3 * 10, max_plot_counter * 10))

                for img_index in img_indexes:
                    tr_image = train_dataset[img_index][0].to(device)
                    tst_image = test_dataset[img_index][0].to(device)

                    output = model(tr_image)

                    outputs.append(output)
                    test_images.append(tst_image)

                    if plot_train_image_flag:
                        if plot_counter < max_plot_counter:
                            train_grayscale_image = tr_image.clone()
                            nn_image = output.clone()
                            original_image = tst_image.clone()

                            fig_train.add_subplot(max_plot_counter, 3, plot_counter * 3 + 1)
                            plt.tight_layout()
                            plt.axis('off')
                            plt.title('Grayscale')
                            plt.imshow(train_grayscale_image.cpu().detach().numpy().transpose((1, 2, 0)), cmap='gray')

                            fig_train.add_subplot(max_plot_counter, 3, plot_counter * 3 + 2)
                            plt.tight_layout()
                            plt.axis('off')
                            plt.title('NN version')
                            plt.imshow(nn_image.cpu().detach().numpy().transpose((1, 2, 0)), cmap='gray')

                            fig_train.add_subplot(max_plot_counter, 3, plot_counter * 3 + 3)
                            plt.tight_layout()
                            plt.axis('off')
                            plt.title('Original')
                            plt.imshow(original_image.cpu().detach().numpy().transpose((1, 2, 0)), cmap='gray')

                            plot_counter += 1

                if plot_train_image_flag:
                    plt.title(f"Epoch {epoch}, batch {k}")
                    plt.show()

                outputs = torch.stack(outputs, dim=0)
                test_images = torch.stack(test_images, dim=0)

                loss = loss_fn(outputs, test_images)
                loss.backward()

                optimizer.step()

                train_loss += loss.item()
                train_loop.set_postfix({"loss": loss.item()})

            if writer:
                writer.add_scalar("Loss/train", train_loss / len(train_loader), epoch)

            with torch.no_grad():
                model.eval()
                val_loop = tqdm(enumerate(val_loader, 0), total=len(val_loader), desc="Val")
                for j, (img_indexes) in val_loop:

                    outputs = []
                    test_images = []

                    plot_counter = 0
                    max_plot_counter = 3

                    fig_val = plt.figure(figsize=(3 * 10, max_plot_counter * 10))

                    for img_index in img_indexes:
                        train_val_image = train_dataset[img_index][0].to(device)
                        test_val_image = test_dataset[img_index][0].to(device)

                        output = model(train_val_image)

                        if plot_val_image_flag:
                            if plot_counter < max_plot_counter :
                                plot_counter += 1

                                val_grayscale_image = train_val_image.clone()
                                nn_image = output.clone()
                                original_image = test_val_image.clone()

                                fig_val.add_subplot(max_plot_counter, 3, plot_counter * 3 + 1)
                                plt.tight_layout()
                                plt.axis('off')
                                plt.title('Grayscale')
                                plt.imshow(val_grayscale_image.cpu().detach().numpy().transpose((1, 2, 0)), cmap='gray')

                                fig_val.add_subplot(max_plot_counter, 3, plot_counter * 3 + 2)
                                plt.tight_layout()
                                plt.axis('off')
                                plt.title('NN version')
                                plt.imshow(nn_image.cpu().numpy().transpose((1, 2, 0)), cmap='gray')

                                fig_val.add_subplot(max_plot_counter, 3, plot_counter * 3 + 3)
                                plt.tight_layout()
                                plt.axis('off')
                                plt.title('Original')
                                plt.imshow(original_image.cpu().numpy().transpose((1, 2, 0)), cmap='gray')

                        outputs.append(output)
                        test_images.append(test_val_image)

                    if plot_val_image_flag:
                        plt.title(f"Epoch {epoch}, batch {j}")
                        plt.show()

                    outputs = torch.stack(outputs, dim=0)
                    test_images = torch.stack(test_images, dim=0)

                    acc = 1 - loss_fn(outputs, test_images).item()

                    val_loop.set_postfix({"acc": acc})

                if acc > best:
                    torch.save(model.state_dict(), ckpt_path)
                    best = acc


    print("[*] Model created.")

    train(model, optimizer, loss_fn, train_dataloader, val_dataloader, epochs=num_epochs)
