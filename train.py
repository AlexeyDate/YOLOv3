import argparse
from terminaltables import AsciiTable

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
parser.add_argument('--image_size', type=int, default=416, help='Input image size')
parser.add_argument('--weights', type=str, default=None,
                    help='Path to Darknet53 weight file or path to YOLOv3 weight file')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
parser.add_argument('--epochs', type=int, default=100, help='Total epochs')
parser.add_argument('--lr', type=float, default=0.00007, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0005, help='Weight decay')
parser.add_argument('--multiscale', action='store_true', default=False, help='Enable multi-scale training')
args = parser.parse_args()

anchors = [[[313, 303], [336, 323], [306, 371]],
           [[139, 222], [149, 212], [171, 205]],
           [[23, 34], [99, 234], [129, 224]]]

train_dataset = Dataset(
    data_dir=data_train,
    labels_dir=data_label,
    anchors=anchors,
    image_size=args.image_size,
    num_classes=classes,
    file_format=file_format,
    type_dataset='train',
    convert_to_yolo=convert_to_yolo
)

val_dataset = Dataset(
    data_dir=data_test,
    labels_dir=data_label,
    anchors=anchors,
    image_size=args.image_size,
    num_classes=classes,
    file_format=file_format,
    type_dataset='validation',
    convert_to_yolo=convert_to_yolo
)

# Few checks to make sure the solution is correct
assert isinstance(train_dataset[0], dict)
assert len(train_dataset[0]) == 3
assert isinstance(train_dataset[0]['image'], torch.Tensor)
assert isinstance(train_dataset[0]['converted_target'], list)
assert isinstance(train_dataset[0]['target'], torch.Tensor)

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

if args.weights.endswith('.weights'):
    model = YOLOv3(anchors=anchors, num_classes=classes, darknet_weights=args.weights).to(device)
elif args.weights.endswith('.pt'):
    model = YOLOv3(anchors=anchors, num_classes=classes).to(device)
    model.load_state_dict(torch.load(args.weights))
else:
    model = YOLOv3(anchors=anchors, num_classes=classes).to(device)

loss = YOLOLoss(anchors=anchors).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.98)

print(AsciiTable(
    [
        ["Parameter", "value"],
        ["epochs", args.epochs],
        ["batch size", args.batch_size],
        ["learning rate", args.lr],
        ["weight decay", args.weight_decay],
        ["image size", f'{args.image_size}x{args.image_size}'],
        ["multiscale", "on" if args.multiscale else "off"]
    ]).table)

fit(model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    criterion=loss,
    epochs=args.epochs,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    train_dataset=train_dataset if args.multiscale else None,
    backup=backup,
    device=device,
    verbose=True)

if args.epochs > 0:
    torch.save(model.state_dict(), backup + 'yolov3_' + str(args.epochs) + '.pt')

pred_boxes, true_boxes = get_bound_boxes(train_dataloader, model, anchors, image_size=args.image_size, nms_threshold=0.4, threshold=0.1, device=device)
mAP = mean_average_precision(pred_boxes, true_boxes, classes=classes, iou_threshold=0.5)
print(f'Train mAP: {mAP}')

pred_boxes, true_boxes = get_bound_boxes(val_dataloader, model, anchors, image_size=args.image_size, nms_threshold=0.4, threshold=0.1, device=device)
mAP = mean_average_precision(pred_boxes, true_boxes, classes=classes, iou_threshold=0.5)
print(f'Validation mAP: {mAP}')
