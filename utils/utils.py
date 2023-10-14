import math
import torch
from tqdm import tqdm


def intersection_over_union(predicted_bbox, ground_truth_bbox, GIoU=False, DIoU=False, CIoU=False) -> torch.tensor:
    """
    Intersection Over Union for 2 rectangles.
    You can read the information about all metrics on the follow links:
    https://learnopencv.com/iou-loss-functions-object-detection/

    param: predicted_bbox - predicted bboxes
    param: ground_truth_bbox - target tbboxes

    return: Intersection Over Union tensor

    Note: be careful with tensor views.
    Epected predicted_bbox view: (x, y, w, h)
    Expected ground_truth_bbox view: (x, y, w, h)
    """

    # Predicted and target values in standard YOLO format.
    # Converting values to x1, y1, x2, y2

    predicted_bbox_x1 = predicted_bbox[..., 0] - predicted_bbox[..., 2] / 2
    predicted_bbox_x2 = predicted_bbox[..., 0] + predicted_bbox[..., 2] / 2
    predicted_bbox_y1 = predicted_bbox[..., 1] - predicted_bbox[..., 3] / 2
    predicted_bbox_y2 = predicted_bbox[..., 1] + predicted_bbox[..., 3] / 2

    ground_truth_bbox_x1 = ground_truth_bbox[..., 0] - ground_truth_bbox[..., 2] / 2
    ground_truth_bbox_x2 = ground_truth_bbox[..., 0] + ground_truth_bbox[..., 2] / 2
    ground_truth_bbox_y1 = ground_truth_bbox[..., 1] - ground_truth_bbox[..., 3] / 2
    ground_truth_bbox_y2 = ground_truth_bbox[..., 1] + ground_truth_bbox[..., 3] / 2

    intersection_x1 = torch.max(predicted_bbox_x1, ground_truth_bbox_x1)
    intersection_x2 = torch.min(predicted_bbox_x2, ground_truth_bbox_x2)
    intersection_y1 = torch.max(predicted_bbox_y1, ground_truth_bbox_y1)
    intersection_y2 = torch.min(predicted_bbox_y2, ground_truth_bbox_y2)

    zeros = torch.zeros_like(predicted_bbox_x1)
    intersection_area = torch.max(intersection_x2 - intersection_x1, zeros) * torch.max(
        intersection_y2 - intersection_y1, zeros
    )

    eps=1e-9

    predicted_w = predicted_bbox_x2 - predicted_bbox_x1
    predicted_h = predicted_bbox_y2 - predicted_bbox_y1
    area_predicted = predicted_w * predicted_h

    ground_truth_w = ground_truth_bbox_x2 - ground_truth_bbox_x1
    ground_truth_h = ground_truth_bbox_y2 - ground_truth_bbox_y1
    area_gt = ground_truth_w * ground_truth_h

    union_area = area_predicted + area_gt - intersection_area

    iou = intersection_area / (union_area + eps)

    if GIoU or DIoU or CIoU:
        # Convex (smallest enclosing box)
        cw = torch.max(predicted_bbox_x2, ground_truth_bbox_x2) - torch.min(predicted_bbox_x1, ground_truth_bbox_x1)
        ch = torch.max(predicted_bbox_y2, ground_truth_bbox_y2) - torch.min(predicted_bbox_y1, ground_truth_bbox_y1)
      
        if GIoU:
            c_area = cw * ch
            iou = iou - ((c_area - union_area) / (c_area + eps))
      
        elif DIoU or CIoU:
            c2 = cw ** 2 + ch ** 2 # Convex diagonal squared
            dw = (predicted_bbox_x1 + predicted_bbox_x2 - ground_truth_bbox_x1 - ground_truth_bbox_x2) / 2
            dh = (predicted_bbox_y1 + predicted_bbox_y2 - ground_truth_bbox_y1 - ground_truth_bbox_y2) / 2
            d2 = dw ** 2 + dh ** 2 # Center distance squared
            
            if DIoU:
                iou = iou - (d2 / (c2 + eps))
          
            elif CIoU:
              v = (4 / math.pi ** 2) * \
                  torch.pow((torch.atan(ground_truth_w / ground_truth_h) - torch.atan(predicted_w / predicted_h)), 2)
              with torch.no_grad():
                  alpha = v / ((1 + eps) - iou + v)

              iou = iou - (d2 / c2 + v * alpha)       

    return iou


def inersection_over_union_anchors(bbox_wh, anchors) -> torch.tensor:
    """
    Intersection Over Union for boundig box with anchor boxes with same centers.

    param: bbox_wh - bounding box width and height
    param: anchors - anchor box width and height

    return: Intersection Over Union tensor
    """
    w1, h1 = bbox_wh
    num_anchors = anchors.size(0)
    iou = torch.empty(num_anchors, dtype=torch.float32)
    for i in range(num_anchors):
        w2, h2 = anchors[i]
        intersection_area = min(w1, w2) * min(h1, h2)
        union_area = (w1 * h1) + (w2 * h2) - intersection_area
        iou[i] = intersection_area / union_area

    return iou


def non_max_suppression(bboxes, iou_threshold):
    """
    Non-Maximum Suppression.

    param: bboxes - all predicted bounding boxes with valid confidience values
    param: iou_threshold - intesection over union threshold

    return: correct bounding boxes
    """
    bboxes = sorted(bboxes, key=lambda x: x[4], reverse=True)
    non_max_bboxes = []
    while bboxes:
        current_box = bboxes.pop(0)
        non_max_bboxes.append(current_box)

        temp_bboxes = []
        for box in bboxes:
            class_box = torch.argmax(box[5:])
            class_current_box = torch.argmax(current_box[5:])

            if intersection_over_union(current_box[:4], box[:4]).item() < iou_threshold or class_box != class_current_box:
                temp_bboxes.append(box)
        bboxes = temp_bboxes

    return non_max_bboxes


def convert_to_yolo(bbox, image_size, anchors, s) -> torch.Tensor:
    """
    Converting predicted coordinates to standard YOLO format.

    param: bbox - bounding boxes
    param: image_size - input image size
    param: anchors - anchor boxes
    param: s - current grid size

    return: calculated all bounding boxes in standard YOLO format

    Note: tx, ty relative to grid cell convert to relative to image.
    tw, th convert to bw, bh by using anchor boxes
    """

    device = bbox.device
    anchors = anchors.to(device)
    
    grid_y, grid_x = torch.meshgrid(torch.arange(s), torch.arange(s), indexing='ij')
    grid_y = grid_y.contiguous().view(1, s, s, 1).to(device)
    grid_x = grid_x.contiguous().view(1, s, s, 1).to(device)

    # Original method
    bbox[..., 0] = (bbox[..., 0].sigmoid() + grid_x) * (image_size / s)
    bbox[..., 1] = (bbox[..., 1].sigmoid() + grid_y) * (image_size / s)
    bbox[..., 2:4] = torch.exp(bbox[..., 2:4]) * anchors

    # Power method
    # bbox[..., 0] = (bbox[..., 0].sigmoid() * 2. - 0.5 + grid_x) * (input_size / s)
    # bbox[..., 1] = (bbox[..., 1].sigmoid() * 2. - 0.5 + grid_y) * (input_size / s)
    # bbox[..., 2:4] = (bbox[..., 2:4].sigmoid() * 2) ** 2 * anchors

    bbox[..., 4:] = bbox[..., 4:].sigmoid()

    return bbox

def get_bound_boxes(loader, model, anchors, image_size=416, nms_threshold=0.5, threshold=0.1, device='cpu'):
    """
    Getting predicted and target bounding boxes with Non-Maximum Supression.

    param: loader - dataloader
    param: model - model
    param: anchors - anchor boxes
    param: image_size - input image size
    param: nms_threshold - Intersection Over Union threshold (default = 0.5) for NMS
    param: threshold - confidience threshold (default = 0.1)
    param: device - device of model (default = cpu)

    return: all prediction bounding boxes, all true bounding boxes
    """

    assert isinstance(loader, torch.utils.data.dataloader.DataLoader), \
        "loader does not match the type of torch.utils.data.dataloader.DataLoader"

    # Setting values
    anchors = torch.tensor(anchors, dtype=torch.float32)
    all_pred_boxes = []
    all_true_boxes = []

    model.eval()
    for i, batch in enumerate(tqdm(loader, desc=f'Prediction all bounding boxes', leave=False)):
        images = batch['image'].to(device)
        targets = batch['target'].to(device)
        
        with torch.no_grad():
            predictions = model(images)

        batch_size = predictions[0].size(0) 

        # Concatenating results of each scale and apply non-maximum suppression
        for j in range(batch_size):
            for scale_index in range(len(predictions)):
                s = predictions[scale_index][0].size(1)
                anchors_per_scale = anchors[scale_index]

                # Converting predictions to standard YOLO format
                predicted_bbox = convert_to_yolo(predictions[scale_index][j], image_size, anchors_per_scale, s)

                mask_pred = predicted_bbox[..., 4] >= threshold
                if scale_index == 0:
                    image_pred_bboxes = predicted_bbox[mask_pred, :]   
                else:
                    image_pred_bboxes = torch.cat((image_pred_bboxes, predicted_bbox[mask_pred, :]), dim=0)

            image_pred_bboxes = non_max_suppression(image_pred_bboxes, nms_threshold)
            all_pred_boxes.append(image_pred_bboxes)

            image_true_bboxes = targets[j, targets[j, :, 0] > -1]
            image_true_bboxes[:, :4] *= image_size
            all_true_boxes.append(image_true_bboxes)

    return all_pred_boxes, all_true_boxes



