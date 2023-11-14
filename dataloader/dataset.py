import os
from albumentations.core.bbox_utils import calculate_bbox_area
import xmltodict
import torch
import numpy as np
import albumentations as alb
import albumentations.pytorch
from PIL import Image
from torch.utils.data import Dataset
from utils.utils import inersection_over_union_anchors

class Dataset(Dataset):
    """
    Base class of txt and xml description files.
    Class can convert [x1, y1, x2, y2] view to the YOLO standart format
    and create a target matrix after creating dataloader

    Note: shape of target = (batch size, grid size, grid size, num_anchors, 5 + num_classes)
    each anchor = [tx, ty, tw, th, t0, c1, c2, ..., cn]
    tx, ty values are calculated relative to the grid cell
    """

    def __init__(self, 
                 data_dir, 
                 labels_dir,
                 anchors,
                 image_size = 416,
                 num_classes=4, 
                 file_format='txt', 
                 type_dataset='train', 
                 convert_to_yolo=True):
        """
        param: data_dir - path to obj.data
        param: labels_dir - path to obj.names
        param: anchors - anchor boxes
        param: image_size - input image size (default = 416)
        param: num_classses - number of classes (default = 4)
        param: file_format - txt or xml format description files (default = 'txt', available = 'xml')
        param: type_dataset - dataset type (default='train', available = 'validation')
        param: convert_to_yolo - needed if the deiscription files have the format of bounding boxes [x1, y1, x2, y2]
        """

        self.class2tag = {}
        with open(labels_dir, 'r') as f:
            for line in f:
                (val, key) = line.split()
                self.class2tag[key] = val

        self.image_paths = []
        self.box_paths = []
        for tag in self.class2tag:
            for file in os.listdir(data_dir + '/' + tag):
                if file.endswith('.jpg'):
                    self.image_paths.append(data_dir + '/' + tag + '/' + file)
                if file.endswith('.' + file_format):
                    self.box_paths.append(data_dir + '/' + tag + '/' + file)

        # Sorting to access values by equivalent files
        self.image_paths = sorted(self.image_paths)
        self.box_paths = sorted(self.box_paths)

        assert len(self.image_paths) == len(self.box_paths)
        assert type_dataset in ['train', 'validation']

        self.input_size = image_size
        self.anchors = torch.tensor(anchors, dtype=torch.float32)
        self.num_anchors = self.anchors.size(0)
        self.num_classes = num_classes
        self.file_format = file_format
        self.type_dataset = type_dataset
        self.convert_to_yolo = convert_to_yolo

        # Variable for multi-scaling 
        self.current_input_size = image_size

    def __getitem__(self, idx):
        image = np.array(Image.open(self.image_paths[idx]).convert("RGB"))

        if self.file_format == 'xml':
            bboxes, class_labels = self.get_boxes_from_xml(self.box_paths[idx])
        if self.file_format == 'txt':
            bboxes, class_labels = self.get_boxes_from_txt(self.box_paths[idx])

        if self.convert_to_yolo:
            for i, box in enumerate(bboxes):
                bboxes[i] = self.convert_to_yolo_box_params(box, image.shape[1], image.shape[0])

        # Creating transformations for training
        if self.type_dataset == 'train':
            transforms = alb.Compose(
                [
                    alb.Resize(self.current_input_size, self.current_input_size),
                    alb.HorizontalFlip(p=0.5),
                    alb.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=(-45, 45), p=0.5),
                    alb.RandomBrightnessContrast(p=0.2),
                    alb.Normalize(),
                    alb.pytorch.ToTensorV2()
                ], bbox_params=alb.BboxParams(format='yolo', label_fields=['class_labels']))

        # Creating transformations for validation
        elif self.type_dataset == 'validation':
            transforms = alb.Compose(
                [
                    alb.Resize(self.input_size, self.input_size),
                    alb.Normalize(),
                    alb.pytorch.ToTensorV2()
                ], bbox_params=alb.BboxParams(format='yolo', label_fields=['class_labels']))

        transformed = transforms(image=image, bboxes=bboxes, class_labels=class_labels)
        transformed_image = transformed['image']
        transformed_bboxes = torch.tensor(transformed['bboxes'])
        transformed_class_labels = torch.tensor(transformed['class_labels'])

        # Target list contains 3 scale and 3 anchor boxes for each scale
        num_anchors_per_scale = 3
        converted_target = []
        grid_size = []
        strides = [32, 16, 8]
        for stride in strides:
            grid_size.append(self.current_input_size // stride)
            converted_target.append(torch.zeros((self.current_input_size // stride, self.current_input_size // stride, num_anchors_per_scale,
                                       self.num_classes + 5), dtype=torch.float32))

        # *****Target will be created in 2 variants:*****
        #
        # 1: Target tensor for training, it is a specific tensor with
        # the following format: (scales, y, x, anchors, num_classes + 5)
        #
        # 2: Target tensor for calculating mAP 
        # in the original yolo format (x, y, w, h, target_class)
        
        # Maximum train object in the image 
        max_objects_per_image = 100
        target = torch.empty(100, 5).fill_(-1)

        for i, box in enumerate(transformed_bboxes):
            
            if i == max_objects_per_scale:
                print('\nThere are too many objects in the image, the number has been cut to 100')
                break

            target[i, :4] = box
            target[i, 4] = transformed_class_labels[i]
            
            for yolo_layer in range(num_anchors_per_scale):
                # Deprecated implementation with iou ignore threshold
                # calculate_ious = intersection_over_union_anchors(bbox_wh=box[2:4], anchors=self.anchors[yolo_layer])

                anchors = self.anchors[yolo_layer] / strides[yolo_layer]
                t = box * grid_size[yolo_layer]
                ratio = t[2:4] / anchors[:, None]
                j = torch.max(ratio, 1. / ratio).max(dim=2)[0] < 4

                x_cell = t[0].long()
                y_cell = t[1].long()
                for anchor_index, true_box in enumerate(j):
                    if true_box:
                        x_cell = x_cell.clamp_(0, grid_size[yolo_layer] - 1)
                        y_cell = y_cell.clamp_(0, grid_size[yolo_layer] - 1)

                        tx = grid_size[yolo_layer] * box[0] - x_cell
                        ty = grid_size[yolo_layer] * box[1] - y_cell

                        converted_target[yolo_layer][y_cell, x_cell, anchor_index, :4] = torch.tensor([tx, ty, t[2], t[3]]) 
                        converted_target[yolo_layer][y_cell, x_cell, anchor_index, 5 + transformed_class_labels[i]] = 1
                        converted_target[yolo_layer][y_cell, x_cell, anchor_index, 4] = 1

        return {"image": transformed_image.float(), "converted_target": converted_target, "target": target}

    def __len__(self):
        return len(self.image_paths)

    def get_boxes_from_txt(self, txt_filename: str):
        boxes = []
        class_labels = []

        with open(txt_filename) as f:
            for obj in f:
                param_list = list(map(float, obj.split()))

                boxes.append(param_list[1:])
                class_labels.append(int(param_list[0]))

        return boxes, class_labels

    def get_boxes_from_xml(self, xml_filename: str):
        boxes = []
        class_labels = []

        with open(xml_filename) as f:
            xml_content = xmltodict.parse(f.read())
        xml_object = xml_content['annotation']['object']

        if type(xml_object) is dict:
            xml_object = [xml_object]

        if type(xml_object) is list:
            for obj in xml_object:
                boxe_list = list(map(float, [obj['bndbox']['xmin'], obj['bndbox']['ymin'], obj['bndbox']['xmax'],
                                             obj['bndbox']['ymax']]))
                boxes.append(boxe_list)
                class_labels.append(self.class2tag[obj['name']])

        return boxes, class_labels

    def convert_to_yolo_box_params(self, box_coordinates, im_w, im_h):
        ans = list()

        ans.append((box_coordinates[0] + box_coordinates[2]) / 2 / im_w)  # x_center
        ans.append((box_coordinates[1] + box_coordinates[3]) / 2 / im_h)  # y_center

        ans.append((box_coordinates[2] - box_coordinates[0]) / im_w)  # width
        ans.append((box_coordinates[3] - box_coordinates[1]) / im_h)  # height
        return ans
