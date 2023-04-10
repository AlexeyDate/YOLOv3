import argparse

import torch
from torch.utils.data import DataLoader

from dataloader.dataset import Dataset
from model.yolo import YOLOv3
from utils.loss import YOLOLoss
from utils.fit import fit
from utils.utils import get_bound_boxes
from utils.mAP import mean_average_precision

data = './data/obj.data'
with open(data, 'r') as f:
    classes = int(f.readline().split()[2])
    data_train = f.readline().split()[2]
    data_test = f.readline().split()[2]
    data_label = f.readline().split()[2]
    backup = f.readline().split()[2]
    file_format = f.readline().split()[2]
    convert_to_yolo = True if f.readline().split()[2] == 'True' else False

parser = argparse.ArgumentParser()
parser.add_argument('--darknet_weights', type=str, default=None, help='Path to Darknet19 weight file')
parser.add_argument('--yolo_weights', type=str, default=None, help='Path to YOLOv2 weight file')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
parser.add_argument('--epochs', type=int, default=100, help='Total epochs')
parser.add_argument('--learning_rate', type=float, default=0.00005, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0002, help='Weight decay')
parser.add_argument('--multiscale_off', action='store_true', default=False, help='Disable multi-scale training')
args = parser.parse_args()

anchors = [[0.16179443, 0.2349594],
           [0.36622464, 0.46251684],
           [0.39050824, 0.77001864],
           [0.68471097, 0.60060138],
           [0.61761504, 0.74928832],
           [0.59013056, 0.90289886],
           [0.78224109, 0.79511188],
           [0.75123652, 0.95919777],
           [0.92122389, 0.88260267]]

train_dataset = Dataset(
    data_dir=data_train,
    labels_dir=data_label,
    anchors=anchors,
    num_classes=classes,
    file_format=file_format,
    type_dataset='train',
    convert_to_yolo=convert_to_yolo
)

val_dataset = Dataset(
    data_dir=data_test,
    labels_dir=data_label,
    anchors=anchors,
    num_classes=classes,
    file_format=file_format,
    type_dataset='validation',
    convert_to_yolo=convert_to_yolo
)

# a few checks to make sure the solution is correct
assert isinstance(train_dataset[0], dict)
assert len(train_dataset[0]) == 2
assert isinstance(train_dataset[0]['image'], torch.Tensor)
assert isinstance(train_dataset[0]['target'], list)

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=args.batch_size,
    shuffle=True)

val_dataloader = DataLoader(
    dataset=val_dataset,
    batch_size=args.batch_size,
    shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Executable device:', device)

if args.darknet_weights is not None:
    model = YOLOv3(anchors=anchors, num_classes=classes, darknet_weights=args.darknet_weights).to(device)
else:
    model = YOLOv3(anchors=anchors, num_classes=classes).to(device)

if args.yolo_weights is not None:
    model.load_state_dict(torch.load(args.yolo_weights))

loss = YOLOLoss(anchors=anchors).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.99)

fit(model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    criterion=loss,
    epochs=args.epochs,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    train_dataset=train_dataset if not args.multiscale_off else None,
    backup=backup,
    device=device,
    verbose=True)

torch.save(model.state_dict(), backup + 'yolov2_' + str(args.epochs) + '.pt')

pred_boxes, true_boxes = get_bound_boxes(train_dataloader, model, anchors, iou_threshold=0.5, threshold=0.45)
mAP = mean_average_precision(pred_boxes, true_boxes, classes=classes, iou_threshold=0.5)
print(f'Train mAP: {mAP}')

pred_boxes, true_boxes = get_bound_boxes(val_dataloader, model, anchors, iou_threshold=0.5, threshold=0.2, device=device)
mAP = mean_average_precision(pred_boxes, true_boxes, classes=classes, iou_threshold=0.5)
print(f'Validation mAP: {mAP}')
