import warnings

import torchvision
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_V2_Weights

from archive.src.config.config import cfg

warnings.filterwarnings(action='ignore')


def build_model(num_classes=cfg.NUM_CLASS + 1):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    return model
