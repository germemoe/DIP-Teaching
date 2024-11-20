import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from cityscapes_dataset import CityscapesDataset
from FCN_network import FullyConvNetwork
from Discriminator import Discriminator
from torch.optim.lr_scheduler import StepLR

def tensor_to_image(tensor):
    """
    Convert a PyTorch tensor to a NumPy array suitable for OpenCV.

    Args:
        tensor (torch.Tensor): A tensor of shape (C, H, W).

    Returns:
        numpy.ndarray: An image array of shape (H, W, C) with values in [0, 255] and dtype uint8.
    """
    # Move tensor to CPU, detach from graph, and convert to NumPy array
    image = tensor.cpu().detach().numpy()
    # Transpose from (C, H, W) to (H, W, C)
    image = np.transpose(image, (1, 2, 0))
    # Denormalize from [-1, 1] to [0, 1]
    image = (image + 1) / 2
    # Scale to [0, 255] and convert to uint8
    image = (image * 255).astype(np.uint8)
    return image

def save_images(inputs, targets, outputs, folder_name, epoch, num_images=5):
    """
    Save a set of input, target, and output images for visualization.

    Args:
        inputs (torch.Tensor): Batch of input images.
        targets (torch.Tensor): Batch of target images.
        outputs (torch.Tensor): Batch of output images from the model.
        folder_name (str): Directory to save the images ('train_results' or 'val_results').
        epoch (int): Current epoch number.
        num_images (int): Number of images to save from the batch.
    """
    os.makedirs(f'{folder_name}/epoch_{epoch}', exist_ok=True)
    for i in range(num_images):
        # Convert tensors to images
        input_img_np = tensor_to_image(inputs[i])
        target_img_np = tensor_to_image(targets[i])
        output_img_np = tensor_to_image(outputs[i])

        # Concatenate the images horizontally
        comparison = np.hstack((input_img_np, target_img_np, output_img_np))

        # Save the comparison image
        cv2.imwrite(f'{folder_name}/epoch_{epoch}/result_{i + 1}.png', comparison)

def train_one_epoch(dis_model, gen_model, dataloader, dis_optimizer, gen_optimizer, criterion, device, epoch, num_epochs):
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for the training data.
        optimizer (Optimizer): Optimizer for updating model parameters.
        criterion (Loss): Loss function.
        device (torch.device): Device to run the training on.
        epoch (int): Current epoch number.
        num_epochs (int): Total number of epochs.
    """
    dis_model.train()
    gen_model.train()
    LAMBDA = 100
    for i, (image_rgb, image_semantic) in enumerate(dataloader):
        # Move data to the device
        image_rgb = image_rgb.to(device)
        image_semantic = image_semantic.to(device)

        # Zero the gradients
        dis_optimizer.zero_grad()

        # Forward pass
        dis_real_outputs = dis_model(image_rgb, image_semantic)
        gen_outputs = gen_model(image_rgb)
        dis_fake_outputs = dis_model(image_rgb, gen_outputs.detach())

        # Save sample images every 5 epochs
        if epoch % 5 == 0 and i == 0:
            save_images(image_rgb, image_semantic, gen_outputs, 'train_results', epoch)

        # Compute the loss
            
        dis_real_loss = criterion(dis_real_outputs, torch.ones_like(dis_real_outputs, device=device))
        dis_fake_loss = criterion(dis_fake_outputs, torch.zeros_like(dis_fake_outputs, device=device))
        dis_loss = dis_real_loss + dis_fake_loss

        # Backward pass and optimization
        dis_loss.backward()
        dis_optimizer.step()


        # Zero the gradients
        gen_optimizer.zero_grad()

        # Compute the loss        
        dis_fake_outputs_with_grad = dis_model(image_rgb, gen_outputs)
        gen_loss_crossentropyloss = criterion(dis_fake_outputs_with_grad, torch.ones_like(dis_fake_outputs, device=device))
        gen_l1_loss = torch.mean(torch.abs(gen_outputs - image_semantic)) 
        gen_loss = gen_loss_crossentropyloss + LAMBDA * gen_l1_loss

        # Backward pass and optimization
        gen_loss.backward()
        gen_optimizer.step()

        # Print loss information
        print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {dis_loss.item():.4f}')
        print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {gen_loss.item():.4f}')

def validate(dis_model, gen_model, dataloader, criterion, device, epoch, num_epochs):
    """
    Validate the model on the validation dataset.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for the validation data.
        criterion (Loss): Loss function.
        device (torch.device): Device to run the validation on.
        epoch (int): Current epoch number.
        num_epochs (int): Total number of epochs.
    """
    dis_model.eval()
    gen_model.eval()
    val_dis_loss = 0.0
    val_gen_loss = 0.0   
    LAMBDA = 100


    with torch.no_grad():
        for i, (image_rgb, image_semantic) in enumerate(dataloader):
            # Move data to the device
            image_rgb = image_rgb.to(device)
            image_semantic = image_semantic.to(device)

            # Forward pass
            dis_real_outputs = dis_model(image_rgb, image_semantic)
            gen_outputs = gen_model(image_rgb)
            dis_fake_outputs = dis_model(image_rgb, gen_outputs)
            # Compute the loss
            
            dis_real_loss = criterion(dis_real_outputs, torch.ones_like(dis_real_outputs, device=device))
            dis_fake_loss = criterion(dis_fake_outputs, torch.zeros_like(dis_fake_outputs, device=device))
            dis_loss = dis_real_loss + dis_fake_loss

            gen_loss_crossentropyloss = criterion(dis_fake_outputs, torch.ones_like(dis_fake_outputs, device=device))
            gen_l1_loss = torch.mean(torch.abs(gen_outputs - image_semantic)) 
            gen_loss = gen_loss_crossentropyloss + LAMBDA * gen_l1_loss

            val_dis_loss += dis_loss.item()
            val_gen_loss += gen_loss.item()

            # Save sample images every 5 epochs
            if epoch % 5 == 0 and i == 0:
                save_images(image_rgb, image_semantic, gen_outputs, 'val_results', epoch)

    # Calculate average validation loss
    avg_val_dis_loss = val_dis_loss / len(dataloader)
    avg_val_gen_loss = val_gen_loss / len(dataloader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_dis_loss:.4f}')
    print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_gen_loss:.4f}')

def main():
    """
    Main function to set up the training and validation processes.
    """
    # Set device to GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Initialize datasets and dataloaders
    train_dataset = CityscapesDataset(list_file='train_list.txt')
    val_dataset = CityscapesDataset(list_file='val_list.txt')

    train_loader = DataLoader(train_dataset, batch_size=40, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=40, shuffle=False, num_workers=4)

    # Initialize model, loss function, and optimizer
    gen_model = FullyConvNetwork().to(device)
    dis_model = Discriminator().to(device)
    gen_model = nn.DataParallel(gen_model)
    dis_model = nn.DataParallel(dis_model)
    criterion = nn.BCELoss()
    dis_optimizer = optim.Adam(dis_model.parameters(), lr=0.00001, betas=(0.5, 0.999))
    gen_optimizer = optim.Adam(gen_model.parameters(), lr=0.0001, betas=(0.5, 0.999))
    # Add a learning rate scheduler for decay
    dis_scheduler = StepLR(dis_optimizer, step_size=200, gamma=0.2)
    gen_scheduler = StepLR(gen_optimizer, step_size=200, gamma=0.2)
    # Training loop
    num_epochs = 800
    for epoch in range(num_epochs):
        train_one_epoch(dis_model, gen_model, train_loader, dis_optimizer, gen_optimizer, criterion, device, epoch, num_epochs)
        validate(dis_model, gen_model, val_loader, criterion, device, epoch, num_epochs)

        # Step the scheduler after each epoch
        dis_scheduler.step()
        gen_scheduler.step()

        # Save model checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(dis_model.state_dict(), f'checkpoints/pix2pix_dis_model_epoch_{epoch + 1}.pth')
            torch.save(gen_model.state_dict(), f'checkpoints/pix2pix_gen_model_epoch_{epoch + 1}.pth')
if __name__ == '__main__':

    main()
    