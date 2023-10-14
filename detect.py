import argparse
import time
import random

import torch
import cv2
import albumentations as alb
import albumentations.pytorch
from model.yolo import YOLOv3
from utils.utils import non_max_suppression
from utils.utils import convert_to_yolo

data = './data/obj.data'
with open(data, 'r') as f:
    classes = int(f.readline().split()[2])
    f.readline()
    f.readline()
    data_label = f.readline().split()[2]
    backup = f.readline().split()[2]

parser = argparse.ArgumentParser()
parser.add_argument('--data_test', type=str, default=None, help='Testing data')
parser.add_argument('--weights', type=str, default=None, help='Path to YOLOv2 weight file')
parser.add_argument('--output', type=str, default=None, help='Path to save output file')
parser.add_argument('--video', action='store_true', default=False, help='Enable object detection on video')
parser.add_argument('--show', action='store_true', default=False, help='Show image or video during object detection')
args = parser.parse_args()

threshold = 0.3
anchors = [[0.16311909, 0.18589602],
           [0.22665434, 0.38434604],
           [0.37126869, 0.52367880],
           [0.35679135, 0.77865950],
           [0.57373131, 0.54419005],
           [0.50775443, 0.83491387],
           [0.78568474, 0.70004242],
           [0.65448186, 0.86748236],
           [0.87333577, 0.90867049]]


transform = alb.Compose(
     [
        alb.Resize(416, 416),
        alb.Normalize(),
        alb.pytorch.ToTensorV2()
     ])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Executable device:', device)

model = YOLOv3(anchors=anchors, num_classes=classes).to(device)
try:
    model.load_state_dict(torch.load(args.weights))
except Exception:
    print('WeightLoadingError: please check your PyTorch weight file')
    exit(-1)

cap = cv2.VideoCapture(args.data_test)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if not cap.isOpened():
    print("DataLoadingError: please check file path")
    exit(-1)

if args.output and args.video:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(args.output, fourcc, 30.0, (width, height))

anchors = torch.tensor(anchors, dtype=torch.float32)
frame_counter = 0
start = time.time()
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        transformed = transform(image=frame)
        transformed_image = transformed["image"]

        transformed_image = transformed_image.unsqueeze(dim=0)
        transformed_image = transformed_image.to(device)

        model.eval()
        with torch.no_grad():
            predictions = model(transformed_image)

        num_anchors_per_scale = predictions[0].size(3)
        for scale_index in range(len(predictions)):
            s = predictions[scale_index][0].size(1)
            anchors_per_scale = anchors[
                                scale_index * num_anchors_per_scale:scale_index * num_anchors_per_scale + num_anchors_per_scale]

            # convert predictions to standard YOLO format
            predicted_bbox = convert_to_yolo(predictions[scale_index], anchors_per_scale, s, with_softmax=True)

            mask_pred = predicted_bbox[..., 0] >= threshold
            if scale_index == 0:
                all_pred_bboxes = predicted_bbox[mask_pred, :]
            else:
                all_pred_bboxes = torch.cat((all_pred_bboxes, predicted_bbox[mask_pred, :]), dim=0)

        predicted_bbox = non_max_suppression(all_pred_bboxes, iou_threshold=0.5)

        labels = [[str, tuple] for i in range(classes)]
        colors = [(0, 0, 255), (0, 165, 255), (0, 255, 255), (100, 255, 40)]
        with open(data_label, 'r') as f:
            for line in f:
                (val, key) = line.split()
                labels[int(val)][0] = key

                if int(val) < len(colors):
                    labels[int(val)][1] = colors[int(val)]
                else:
                    labels[int(val)][1] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        height, width, _ = frame.shape
        for box in predicted_bbox:
            conf = box[0].item()
            x1 = int(box[1] * width - box[3] * width / 2)
            y1 = int(box[2] * height - box[4] * height / 2)
            x2 = int(box[1] * width + box[3] * width / 2)
            y2 = int(box[2] * height + box[4] * height / 2)
            choose_class = torch.argmax(box[5:])

            line_thickness = 2
            text = labels[choose_class][0] + ' ' + str(round(conf, 2))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color=labels[choose_class][1], thickness=line_thickness)
            size, baseline = cv2.getTextSize(text, cv2.FONT_ITALIC, fontScale=0.5, thickness=1)
            text_w, text_h = size
            cv2.rectangle(frame, (x1, y1), (x1 + text_w + line_thickness, y1 + text_h + baseline),
                          color=labels[choose_class][1], thickness=-1)
            cv2.putText(frame, text, (x1 + line_thickness, y1 + 2 * baseline + line_thickness), cv2.FONT_ITALIC,
                        fontScale=0.5, color=(0, 0, 0), thickness=1, lineType=9)

        if args.show:
            cv2.imshow('Detect', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        if args.output:
            if not args.video:
                cv2.imwrite(args.output, frame)
            else:
                out_video.write(frame)
        if args.video:
            frame_counter += 1
            current_time = time.time() - start
            if current_time >= 1:
                print("FPS:", frame_counter)
                start = time.time()
                frame_counter = 0
    else:
        break

if args.output is not None and args.video:
    out_video.release()

if args.show:
    if not args.video:
        cv2.waitKey(0)
    cap.release()
    cv2.destroyWindow('Detect')
