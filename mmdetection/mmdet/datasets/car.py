import os.path as osp

import mmcv
import numpy as np

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset


@DATASETS.register_module(force=True)
class CarDataset(CustomDataset):
    CLASSES = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
               '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
               '20', '21', '22', '23', '24', '25', '26', '27', '28', '29',
               '30', '31', '32', '33')

    def load_annotations(self, ann_file):
        # load image list from file
        image_list = mmcv.list_from_file(self.ann_file)

        self.data_infos = []
        # convert annotations to middle format
        for image_id in image_list:
            filename = f'{self.img_prefix}/{image_id}.png'
            image = mmcv.imread(filename)
            height, width = image.shape[:2]

            data_info = dict(filename=f'{image_id}.png', width=width, height=height)

            # load annotations
            label_prefix = self.img_prefix

            if self.test_mode == False:

                with open(osp.join(label_prefix, f'{image_id}.txt'), 'r') as file:
                    lines = file.readlines()
                    bbox_names = []
                    bboxes = []
                    for line in lines:
                        values = list(map(float, line.strip().split(' ')))
                        class_id = int(values[0])
                        x_min, y_min = int(round(values[1])), int(round(values[2]))
                        x_max, y_max = int(round(max(values[3], values[5], values[7]))), int(
                            round(max(values[4], values[6], values[8])))
                        bbox_names.append(class_id)
                        bboxes.append([x_min, y_min, x_max, y_max])

                gt_bboxes = []
                gt_labels = []
                gt_bboxes_ignore = []
                gt_labels_ignore = []

                # filter 'DontCare'
                for bbox_name, bbox in zip(bbox_names, bboxes):
                    gt_labels.append(bbox_name)
                    gt_bboxes.append(bbox)

                data_anno = dict(
                    bboxes=np.array(gt_bboxes, dtype=np.float32).reshape(-1, 4),
                    labels=np.array(gt_labels, dtype=np.int64),
                    bboxes_ignore=np.array(gt_bboxes_ignore,
                                           dtype=np.float32).reshape(-1, 4),
                    labels_ignore=np.array(gt_labels_ignore, dtype=np.int64))

                data_info.update(ann=data_anno)
                self.data_infos.append(data_info)
            else:
                self.data_infos.append(data_info)

        return self.data_infos

    def get_anno_info(self, idx):
        return self.data_infos[idx]['ann']
