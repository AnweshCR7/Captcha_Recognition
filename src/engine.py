from tqdm import tqdm
import torch
import config
import model_utils


def train_fn(model, data_loader, optimizer, save_model=False):
    model.train()
    fin_loss = 0
    tk = tqdm(data_loader, total=len(data_loader))
    for data in tk:
        for key, value in data.items():
            data[key] = value.to(config.DEVICE)
        optimizer.zero_grad()
        _, loss = model(**data)
        loss.backward()
        optimizer.step()
        fin_loss += loss.item()

    if save_model:
        model_utils.save_model(model, optimizer, loss, config.CHECKPOINT_PATH)

    return fin_loss/len(data_loader)


def eval_fn(model, data_loader):
    model.eval()
    fin_loss = 0
    fin_preds = []
    with torch.no_grad():
        tk = tqdm(data_loader, total=len(data_loader))
        for data in tk:
            for key, value in data.items():
                data[key] = value.to(config.DEVICE)
            batch_preds, loss = model(**data)
            fin_loss += loss.item()
            fin_preds.append(batch_preds)
        return fin_preds, fin_loss / len(data_loader)