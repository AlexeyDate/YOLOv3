import torch
from torch import nn
from utils.utils import intersection_over_union

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
        Expected target view: 3 scale list with each view equals (batch size, s, s, num anchors, (5 + num classes))
        Epected predicted view: 3 scale list with each view equals (batch size, s, s, num anchors, (5 + num classes))
        """
        self.BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0]))
        self.BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0]))

        self.anchors = torch.tensor(anchors, dtype=torch.float32)
        self.num_anchors = self.anchors.size(0)
        self.strides = [32, 16, 8]

    def forward(self, predictions, targets):
        device = predictions[0].device
        box_loss, obj_loss, classes_loss = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)

        # Calculation the loss at the level of the corresponding scale.
        # Note: expected features should be equals in shape between predictions and targets:
        # *****First  scale*****  -> large features
        # *****Second scale*****  -> medium features
        # *****Third  scale*****  -> small features
        for layer_index, (layer_predictions, layer_targets) in enumerate(zip(predictions, targets)):    
            # Build empty object target tensor to calculate object loss
            tobj = torch.zeros_like(layer_predictions[..., 0], device=layer_predictions.device)

            # Creation the mask for target positions
            target_mask = layer_targets[..., 4] == 1

            # Calculation targets per batch
            num_targets = target_mask.count_nonzero()

            # Checking for the presence of targets for this batch
            if num_targets:
                num_anchors_per_scale = predictions[0].size(3)

                # Target and predicted anchor box values:
                # Target:     [conf, tx, ty, tw, th, c1, c2, ..., cn]
                # Prediction: [conf, tx, ty, tw, th, c1, c2, ..., cn]

                # Anchor boxes needed to convert values to prediction boxes
                anchors = (self.anchors[layer_index] / self.strides[layer_index]).to(device)

                pbox = layer_predictions
                tbox = layer_targets

                # Original method
                pbox[..., 2:4] = torch.exp(pbox[..., 2:4]) * anchors

                # Power method
                # pbox[..., :2]  = pbox[..., :2].sigmoid() * 2. - 0.5 
                # pbox[..., 2:4] = (pbox[..., 2:4].sigmoid() * 2) ** 2 * anchors

                pbox = pbox[target_mask]
                tbox = tbox[target_mask]

                # Calculate IoU between predictions and targets.
                # You may use different metrics: CIoU, DIoU, GIoU, IoU
                # *****CIoU recommended*****
                iou = intersection_over_union(pbox[:, :4], tbox[:, :4], CIoU=True)

                # Box loss
                box_loss += (1.0 - iou).mean()

                # Classes loss
                classes_loss += self.BCEcls(pbox[:, 5:], tbox[:, 5:])

                tobj[target_mask] = iou.detach().clamp(0)

            # Object loss
            obj_loss += self.BCEobj(layer_predictions[..., 4], tobj)

        # Using balancing values 
        box_loss *= 0.05
        obj_loss *= 1.0
        classes_loss *= 0.5

        # Expected total_loss propagation
        total_loss = (box_loss + obj_loss + classes_loss)

        return [total_loss, box_loss, obj_loss, classes_loss]