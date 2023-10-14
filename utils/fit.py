import os
import random

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.display import clear_output
from terminaltables import AsciiTable


def evaluate(model, criterion, val_dataloader, device):
    val_loss = [0] * 4
    model.eval()

    for batch in tqdm(val_dataloader, desc=f'Evaluation', leave=False):
        images, targets = batch['image'], batch['converted_target']

        # Moving all target scales to device
        targets = [scale.to(device) for scale in targets]

        # Moving images to device
        images = images.to(device)

        with torch.no_grad():
            predictions = model(images)
            loss = criterion(predictions, targets)
            val_loss = [x.item() + y for x, y in zip(loss, val_loss)]

    return [item / len(val_dataloader) for item in val_loss]


def fit(model, optimizer, scheduler, criterion, epochs, train_dataloader, val_dataloader,
        train_dataset=None, backup='backup/', device='cpu', verbose=False):
    """
    Training model with drawing graph of the loss curve.

    param: model - model to fitting
    param: optimizer - optimizer loss function
    param: scheduler - optimizer scheduler
    param: criterion - loss function
    param: epochs - number of epochs
    param: train_dataloader - dataloader with training split of dataset
    param: val_dataloader - dataloader with validation split of dataset
    param: train_dataset - train dataset for multiscaling training (default = None, which means multiscale is disabled)
    param: backup - path to save loss graph (default = 'backup/')
    param: device - device of model (default = cpu)
    param: verbose - details of loss and resolution (default = False)
    """

    # Creating a directory to save the graph if the directory doesn't exist
    if not os.path.isdir(backup):
        os.mkdir(backup)

    train_log = [[], [], [], []]
    best_train = [float('inf')] * 4

    val_log = [[], [], [], []] 
    best_val = [float('inf')] * 4

    fig = plt.figure(figsize=(9, 7))
    fig_number = fig.number

    # Python standard prefix for color font
    green = '\033[1;32m'
    default = '\033[1;0m'

    if train_dataset and verbose:
        print('Resolution set to', '[', train_dataset.current_input_size, 'x', train_dataset.current_input_size, ']')

    for epoch in range(epochs):
        model.train()
        train_loss = [0] * 4

        for item, batch in enumerate(tqdm(train_dataloader, desc=f"Training, epoch {epoch + 1}", leave=False)):
            images, targets = batch['image'], batch['converted_target']

            # Moving all target scales to device
            targets = [scale.to(device) for scale in targets]
            
            # Moving images to device
            images = images.to(device)

            # Clearing the gradients
            optimizer.zero_grad()

            # Forward pass
            predictions = model(images)
            
            loss = criterion(predictions, targets)
            loss[0].backward()
            train_loss = [x.item() + y for x, y in zip(loss, train_loss)]   
            
            optimizer.step()

            # Every 10 iteration generate a new image scale
            if train_dataset and item % 10 == 9:
                train_dataset.current_input_size = random.randint(10, 19) * 32
                if train_dataset and verbose:
                    print('\nResolution change to', '[', train_dataset.current_input_size, 'x', train_dataset.current_input_size, ']')

        scheduler.step()

        train_loss = [item / len(train_dataloader) for item in train_loss]
        train_log[0].append(train_loss[0])
        train_log[1].append(train_loss[1])
        train_log[2].append(train_loss[2])
        train_log[3].append(train_loss[3])

        val_loss = evaluate(model, criterion, val_dataloader, device)
        val_log[0].append(val_loss[0])
        val_log[1].append(val_loss[1])
        val_log[2].append(val_loss[2])
        val_log[3].append(val_loss[3])

        print('iou_loss:', round(val_log[1][-1], 3))
        print('obj_loss:', round(val_log[2][-1], 3))
        print('class_loss:', round(val_log[3][-1], 3))

        color = [['', ''],
                 ['', ''],
                 ['', ''],
                 ['', '']]
                 
        end = [['', ''],
               ['', ''],
               ['', ''],
               ['', '']]

        if train_log[1][-1] < best_train[1]:
            color[0][0] = green
            end[0][0] = ' ↘' + default
            best_train[1] = train_log[1][-1]

        if train_log[2][-1] < best_train[2]:
            color[1][0] = green   
            end[1][0] = ' ↘' + default
            best_train[2] = train_log[2][-1]

        if train_log[3][-1] < best_train[3]:
            color[2][0] = green
            end[2][0] = ' ↘' + default
            best_train[3] = train_log[3][-1]
        
        if train_log[0][-1] < best_train[0]:
            color[3][0] = green
            end[3][0] = ' ↘' + default
            best_train[0] = train_log[0][-1]

        if val_log[1][-1] < best_val[1]:
            color[0][1] = green
            end[0][1] = ' ↘' + default
            best_val[1] = val_log[1][-1]

        if val_log[2][-1] < best_val[2]:
            color[1][1] = green   
            end[1][1] = ' ↘' + default
            best_val[2] = val_log[2][-1]

        if val_log[3][-1] < best_val[3]:
            color[2][1] = green
            end[2][1] = ' ↘' + default
            best_val[3] = val_log[3][-1]
        
        if val_log[0][-1] < best_val[0]:
            color[3][1] = green
            end[3][1] = ' ↘' + default
            best_val[0] = val_log[0][-1]

        if not plt.fignum_exists(num=fig_number):
            fig = plt.figure(figsize=(9, 7))
            fig_number = fig.number

        clear_output()

        print(f"\nEpoch: {epoch + 1}\{epochs}\n")
        if verbose:
            print(AsciiTable(
                [
                    ["Type",        "Train ",                                                "Validation"],
                    ["Box loss",    color[0][0] + str(round(train_loss[1], 3)) + end[0][0], color[0][1] + str(round(val_loss[1], 3)) + end[0][1]],
                    ["Object loss", color[1][0] + str(round(train_loss[2], 3)) + end[1][0], color[1][1] + str(round(val_loss[2], 3)) + end[1][1]],
                    ["Class loss",  color[2][0] + str(round(train_loss[3], 3)) + end[2][0], color[2][1] + str(round(val_loss[3], 3)) + end[2][1]],
                    ["Total loss",  color[3][0] + str(round(train_loss[0], 3)) + end[3][0], color[3][1] + str(round(val_loss[0], 3)) + end[3][1]]
                ]).table)
        else:
            print(f"train loss: {train_loss[0]}")
            print(f"val loss: {val_loss[0]}")

        steps = list(range(1, epoch + 2))
        plt.subplot(2, 3, 1)
        plt.title("train | box loss")
        plt.plot(steps, train_log[1], marker='o', color='royalblue')
        plt.grid(visible=1, linestyle="--", linewidth=0.5, color="0.5")
        
        plt.subplot(2, 3, 2)
        plt.title("train | obj loss")
        plt.plot(steps, train_log[2], marker='o', color='royalblue')
        plt.grid(visible=1, linestyle="--", linewidth=0.5, color="0.5")

        plt.subplot(2, 3, 3)
        plt.title("train | class loss")
        plt.plot(steps, train_log[3], marker='o', color='royalblue')
        plt.grid(visible=1, linestyle="--", linewidth=0.5, color="0.5")

        plt.subplot(2, 3, 4)
        plt.title("validation | box loss")
        plt.plot(steps, val_log[1], marker='o', color='royalblue')
        plt.grid(visible=1, linestyle="--", linewidth=0.5, color="0.5")
                
        plt.subplot(2, 3, 5)
        plt.title("validation | obj loss")
        plt.plot(steps, val_log[2], marker='o', color='royalblue')
        plt.grid(visible=1, linestyle="--", linewidth=0.5, color="0.5")
                
        plt.subplot(2, 3, 6)
        plt.title("validation | class loss")
        plt.plot(steps, val_log[3], marker='o', color='royalblue')
        plt.grid(visible=1, linestyle="--", linewidth=0.5, color="0.5")

        plt.tight_layout(pad=3.0)
        
        plt.draw()
        plt.pause(0.001)
        fig.savefig(backup + 'loss.png', bbox_inches='tight')
