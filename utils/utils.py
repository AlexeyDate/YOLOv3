import torch
from tqdm import tqdm


def intersection_over_union(predicted_bbox, ground_truth_bbox) -> torch.tensor:
    """
    Intersection Over Union for 2 rectangles.

    param: predicted_bbox - predicted tensor
    param: ground_truth_bbox - target tensor

    return: Intersection Over Union tensor

    Note: be careful with tensor views.
    Epected predicted_bbox view: (batch size, s, s, num anchors, (5 + num classes)
    Expected ground_truth_bbox view: (batch size, s, s, num anchors, 5 + num classes)
    """

    # predicted and target values (standard YOLO format):
    # target:     [conf, x, y, w, h, c1, c2, ..., cn]
    # prediction: [conf, x, y, w, h, c1, c2, ..., cn]

    # convert values to x1, y1, x2, y2
    predicted_bbox_x1 = predicted_bbox[..., 1] - predicted_bbox[..., 3] / 2
    predicted_bbox_x2 = predicted_bbox[..., 1] + predicted_bbox[..., 3] / 2
    predicted_bbox_y1 = predicted_bbox[..., 2] - predicted_bbox[..., 4] / 2
    predicted_bbox_y2 = predicted_bbox[..., 2] + predicted_bbox[..., 4] / 2

    ground_truth_bbox_x1 = ground_truth_bbox[..., 1] - ground_truth_bbox[..., 3] / 2
    ground_truth_bbox_x2 = ground_truth_bbox[..., 1] + ground_truth_bbox[..., 3] / 2
    ground_truth_bbox_y1 = ground_truth_bbox[..., 2] - ground_truth_bbox[..., 4] / 2
    ground_truth_bbox_y2 = ground_truth_bbox[..., 2] + ground_truth_bbox[..., 4] / 2

    intersection_x1 = torch.max(predicted_bbox_x1, ground_truth_bbox_x1)
    intersection_x2 = torch.min(predicted_bbox_x2, ground_truth_bbox_x2)
    intersection_y1 = torch.max(predicted_bbox_y1, ground_truth_bbox_y1)
    intersection_y2 = torch.min(predicted_bbox_y2, ground_truth_bbox_y2)

    zeros = torch.zeros_like(predicted_bbox_x1)
    intersection_area = torch.max(intersection_x2 - intersection_x1, zeros) * torch.max(
        intersection_y2 - intersection_y1, zeros
    )

    area_predicted = (predicted_bbox_x2 - predicted_bbox_x1) * (predicted_bbox_y2 - predicted_bbox_y1)
    area_gt = (ground_truth_bbox_x2 - ground_truth_bbox_x1) * (ground_truth_bbox_y2 - ground_truth_bbox_y1)

    union_area = area_predicted + area_gt - intersection_area

    iou = intersection_area / union_area
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
    bboxes = sorted(bboxes, key=lambda x: x[0], reverse=True)
    non_max_bboxes = []
    while bboxes:
        current_box = bboxes.pop(0)
        non_max_bboxes.append(current_box)

        temp_bboxes = []
        for box in bboxes:
            class_box = torch.argmax(box[5:])
            class_current_box = torch.argmax(current_box[5:])

            if intersection_over_union(current_box, box).item() < iou_threshold or class_box != class_current_box:
                temp_bboxes.append(box)
        bboxes = temp_bboxes

    return non_max_bboxes


def convert_to_yolo(bbox, anchors, s, with_softmax=False) -> torch.Tensor:
    """
    Converting predicted coordinates to standard YOLO format.

    param: bbox - bounding boxes
    param: anchors - anchor boxes
    param: s - current grid size
    param: with_softmax - flag for softmax calculation on inference (default = True)

    return: calculated all bounding boxes in standard YOLO format

    Note: tx, ty relative to grid cell convert to relative to image.
    tw, th convert to bw, bh by using anchor boxes
    """

    device = bbox.device
    anchors = anchors.to(device)

    grid_y, grid_x = torch.meshgrid(torch.arange(s), torch.arange(s), indexing='ij')
    grid_y = grid_y.contiguous().view(1, s, s, 1).to(device)
    grid_x = grid_x.contiguous().view(1, s, s, 1).to(device)

    bbox[..., 1] = (bbox[..., 1] + grid_x) / s
    bbox[..., 2] = (bbox[..., 2] + grid_y) / s

    # original method
    bbox[..., 3] = anchors[:, 0] * torch.exp(bbox[..., 3])
    bbox[..., 4] = anchors[:, 1] * torch.exp(bbox[..., 4])

    # power method
    # bbox[..., 3] = anchors[:, 0] * (2 * bbox[..., 3]) ** 3
    # bbox[..., 4] = anchors[:, 1] * (2 * bbox[..., 4]) ** 3

    if with_softmax:
        bbox[..., 5:] = torch.softmax(bbox[..., 5:], dim=-1)

    return bbox


def get_bound_boxes(loader, model, anchors, nms_threshold=0.5, threshold=0.3, device='cpu'):
    """
    Getting predicted and target bounding boxes with Non-Maximum Supression.

    param: loader - dataloader
    param: model - model
    param: anchors - anchor boxes
    param: nms_threshold - Intersection Over Union threshold (default = 0.5)
    param: threshold - confidience threshold (default = 0.3)
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
        targets = batch['target']
        targets = [scale.to(device) for scale in targets]

        with torch.no_grad():
            predictions = model(images)

        batch_size = predictions[0].size(0)

        # Concatenating results of each scale and apply non-maximum suppression
        for j in range(batch_size):
            for scale_index in range(len(predictions)):
                s = predictions[scale_index][0].size(1)
                num_anchors_per_scale = targets[0].size(3)
                anchors_per_scale = anchors[
                                    scale_index * num_anchors_per_scale:scale_index * num_anchors_per_scale + num_anchors_per_scale]

                # Converting predictions to standard YOLO format
                predicted_bbox = convert_to_yolo(predictions[scale_index][j], anchors_per_scale, s)
                target_bbox = convert_to_yolo(targets[scale_index][j], anchors_per_scale, s)

                mask_pred = predicted_bbox[..., 0] >= threshold
                mask_true = target_bbox[..., 0] == 1
                if scale_index == 0:
                    image_pred_bboxes = predicted_bbox[mask_pred, :]
                    image_true_bboxes = target_bbox[mask_true, :]
                else:
                    image_pred_bboxes = torch.cat((image_pred_bboxes, predicted_bbox[mask_pred, :]), dim=0)
                    image_true_bboxes = torch.cat((image_true_bboxes, target_bbox[mask_true, :]), dim=0)

            image_pred_bboxes = non_max_suppression(image_pred_bboxes, nms_threshold)
            all_pred_boxes.append(image_pred_bboxes)
            all_true_boxes.append(image_true_bboxes)

    return all_pred_boxes, all_true_boxes




