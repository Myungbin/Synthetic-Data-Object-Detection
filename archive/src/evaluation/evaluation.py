import pandas as pd
import torch
from tqdm import tqdm

from archive.src.config.config import cfg


def box_denormalize(x1, y1, x2, y2, width, height):
    x1 = (x1 / cfg.IMG_SIZE) * width
    y1 = (y1 / cfg.IMG_SIZE) * height
    x2 = (x2 / cfg.IMG_SIZE) * width
    y2 = (y2 / cfg.IMG_SIZE) * height
    return x1.item(), y1.item(), x2.item(), y2.item()


def inference(model, test_loader):
    model.eval()
    model.to(cfg.DEVICE)

    results = pd.read_csv('./data/sample_submission.csv')

    for img_files, images, img_width, img_height in tqdm(iter(test_loader)):
        images = [img.to(cfg.DEVICE) for img in images]

        with torch.no_grad():
            outputs = model(images)

        for idx, output in enumerate(outputs):
            boxes = output["boxes"].cpu().numpy()
            labels = output["labels"].cpu().numpy()
            scores = output["scores"].cpu().numpy()

            for box, label, score in zip(boxes, labels, scores):
                x1, y1, x2, y2 = box
                x1, y1, x2, y2 = box_denormalize(x1, y1, x2, y2, img_width[idx], img_height[idx])
                if score > 0.5:
                    results = results.append({
                        "file_name": img_files[idx],
                        "class_id": label - 1,
                        "confidence": score,
                        "point1_x": x1, "point1_y": y1,
                        "point2_x": x2, "point2_y": y1,
                        "point3_x": x2, "point3_y": y2,
                        "point4_x": x1, "point4_y": y2
                    }, ignore_index=True)

    # 결과를 CSV 파일로 저장
    results.to_csv('baseline_submit.csv', index=False)
    print('Done.')
