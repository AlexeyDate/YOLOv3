import torch
from torch import nn
from utils.utils import intersection_over_union
from utils.utils import convert_to_yolo


class YOLOLoss(nn.Module):
    """
    This class represent YOLOv3 loss function.
    All calculations are based on tensors.
    """

    def __init__(self, anchors):
        super().__init__()
        """
        param: anchors - anchor boxes

        Note: be careful with predicted and target views.
        Expected target view: 3 scale list with each view equals (batch size, s, s, num anchors, 5 + num classes)
        Epected predicted view: 3 scale list with each view equals (batch size, s, s, num anchors, (5 + num classes)
        """

        self.mse = nn.MSELoss(reduction='none')
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.bce_conf = nn.BCELoss(reduction='none')

        self.anchors = torch.tensor(anchors, dtype=torch.float32)
        self.num_anchors = self.anchors.size(0)

    def forward(self, predictions, targets):
        xy_loss, wh_loss, obj_loss, no_obj_loss, class_loss = 0, 0, 0, 0, 0

        # Calculation the loss at the level of the corresponding scale.
        # Note: expected features should be equals in shape between predictions and targets:
        # First scale: small features
        # Second scale: medium features
        # Third scale: large features
        for scale_index, (prediction, target) in enumerate(zip(predictions, targets)):
            s = target.size(1)
            batch_size = target.size(0)
            num_anchors_per_scale = target.size(3)

            # target and predicted anchor box values:
            # target:     [conf, tx, ty, tw, th, c1, c2, ..., cn]
            # prediction: [conf, tx, ty, tw, th, c1, c2, ..., cn]

            predict_obj = prediction[..., 0]
            predict_txty = prediction[..., 1:3]
            predict_twth = prediction[..., 3:5]
            predict_classes = prediction[..., 5:]

            target_obj = (target[..., 0] == 1)
            target_noobj = (target[..., 0] == 0)
            target_txty = target[..., 1:3]
            target_twth = target[..., 3:5]
            target_classes = target[..., 5:]

            # Note: loss are calculated only for targets with a confidence of 0 or 1
            #  1 = if object exist with the best anchor
            # -1 = if object exist with not the best anchor (iou > threshold)
            #  0 = others variants (object not exist)

            xy_per_scale_loss = self.mse(predict_txty, target_txty).sum(dim=-1) * target_obj
            xy_loss += xy_per_scale_loss.sum() / batch_size

            wh_per_scale_loss = self.mse(predict_twth, target_twth).sum(dim=-1) * target_obj
            wh_loss += wh_per_scale_loss.sum() / batch_size

            obj_per_scale_loss = self.bce_conf(predict_obj[target_obj], target[..., 0][target_obj])
            obj_loss += (1.0 / batch_size) * obj_per_scale_loss.sum()

            no_obj_per_scale_loss = self.bce_conf(predict_obj[target_noobj], target[..., 0][target_noobj])
            no_obj_loss += (0.2 / batch_size) * no_obj_per_scale_loss.sum()

            class_per_scale_loss = self.bce(predict_classes, target_classes).sum(dim=-1) * target_obj
            class_loss += class_per_scale_loss.sum() / batch_size

        # expected total_loss propagation
        total_loss = xy_loss + wh_loss + obj_loss + no_obj_loss + class_loss

        return [total_loss, xy_loss, wh_loss, obj_loss, no_obj_loss, class_loss]
