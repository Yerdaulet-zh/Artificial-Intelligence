import torch
import torchvision
import numpy as np



def non_max_suppression(bboxes, conf_thres, iou_thres):
    scores = bboxes[0][..., 4]
    bboxes = bboxes[0][scores > conf_thres]
    if len(bboxes):
        scores = bboxes[..., 4]
        bboxes = xywh2xyxy(bboxes[..., :4])
        indxs = torchvision.ops.nms(bboxes, scores, iou_thres)
        return bboxes[indxs]


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


