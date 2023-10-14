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
parser.add_argument('--weights', type=str, default=None,
                    help='Path to Darknet53 weight file or path to YOLOv3 weight file')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
parser.add_argument('--epochs', type=int, default=100, help='Total epochs')
parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0005, help='Weight decay')
parser.add_argument('--multiscale', action='store_true', default=False, help='Enable multi-scale training')
args = parser.parse_args()

anchors = [[0.16311909, 0.18589602],
           [0.22665434, 0.38434604],
           [0.37126869, 0.52367880],
           [0.35679135, 0.77865950],
           [0.57373131, 0.54419005],
           [0.50775443, 0.83491387],
           [0.78568474, 0.70004242],
           [0.65448186, 0.86748236],
           [0.87333577, 0.90867049]]

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

if args.weights.endswith('.weights'):
    model = YOLOv3(anchors=anchors, num_classes=classes, darknet_weights=args.weights).to(device)
elif args.weights.endswith('.pt'):
    model = YOLOv3(anchors=anchors, num_classes=classes).to(device)
    model.load_state_dict(torch.load(args.weights))
else:
    model = YOLOv3(anchors=anchors, num_classes=classes).to(device)

loss = YOLOLoss(anchors=anchors).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.98)

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

pred_boxes, true_boxes = get_bound_boxes(train_dataloader, model, anchors, nms_threshold=0.5, threshold=0.3, device=device)
mAP = mean_average_precision(pred_boxes, true_boxes, classes=classes, iou_threshold=0.5)
print(f'Train mAP: {mAP}')

pred_boxes, true_boxes = get_bound_boxes(val_dataloader, model, anchors, nms_threshold=0.5, threshold=0.3, device=device)
mAP = mean_average_precision(pred_boxes, true_boxes, classes=classes, iou_threshold=0.5)
print(f'Validation mAP: {mAP}')
