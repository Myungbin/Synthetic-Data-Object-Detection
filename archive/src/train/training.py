import numpy as np
from tqdm import tqdm

from archive.src.config.config import cfg


def train(model, train_loader, optimizer, scheduler):
    model.to(cfg.DEVICE)

    best_loss = 9999999
    best_model = None

    for epoch in range(1, cfg.EPOCH + 1):
        model.train()
        train_loss = []
        for images, targets in tqdm(iter(train_loader)):
            images = [img.to(cfg.DEVICE) for img in images]
            targets = [{k: v.to(cfg.DEVICE) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            losses.backward()
            optimizer.step()

            train_loss.append(losses.item())

        if scheduler is not None:
            scheduler.step()

        train_loss = np.mean(train_loss)

        print(f'Epoch [{epoch}] Train loss : [{train_loss:.5f}]\n')

        if best_loss > train_loss:
            best_loss = train_loss
            best_model = model

    return best_model
