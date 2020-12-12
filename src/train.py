import os
import glob
import torch
import numpy as np
from pprint import pprint
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import metrics
import engine
import config
import dataset
from model_utils import plot_loss
from model import CaptchaModel


def remove_duplicates(x):
    if len(x) < 2:
        return x
    fin = ""
    for j in x:
        if fin == "":
            fin = j
        else:
            if j == fin[-1]:
                continue
            else:
                fin = fin + j
    return fin


def decode_predictions(preds, encoder):
    preds = preds.permute(1, 0, 2)
    preds = torch.softmax(preds, 2)
    preds = torch.argmax(preds, 2)
    preds = preds.detach().cpu().numpy()
    cap_preds = []
    for j in range(preds.shape[0]):
        temp = []
        for k in preds[j, :]:
            k = k - 1
            if k == -1:
                temp.append("§")
            else:
                p = encoder.inverse_transform([k])[0]
                temp.append(p)
        tp = "".join(temp).replace("§", "")
        cap_preds.append(remove_duplicates(tp))
    return cap_preds


def run_training():
    image_files = glob.glob(os.path.join(config.DATA_DIR, "*.png"))
    image_files = image_files[:10]
    print(f"Number of Images Found: {len(image_files)}")
    # "../xywz.png" -> "xywz"
    targets_orig = [x.split("/")[-1].split(".")[0] for x in image_files]
    # separate the targets on character level
    targets = [[char for char in x] for x in targets_orig]
    targets_flat = [c for clist in targets for c in clist]

    lbl_encoder = preprocessing.LabelEncoder()
    lbl_encoder.fit(targets_flat)
    targets_enc = [lbl_encoder.transform(x) for x in targets]
    # label encodes from 0, so add 1 to start from 1: 0 will be saved for unknown
    targets_enc = np.array(targets_enc) + 1

    print(f"Number of Unique Classes: {len(lbl_encoder.classes_)}")

    train_imgs, test_imgs, train_targets, test_targets, train_orig_targets, test_orig_targets = \
        model_selection.train_test_split(image_files, targets_enc, targets_orig, test_size=0.1, random_state=42)

    train_dataset = dataset.ClassificationDataset(image_paths=train_imgs, targets=train_targets,
                                                  resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=True
    )

    test_dataset = dataset.ClassificationDataset(image_paths=test_imgs, targets=test_targets,
                                                 resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH))
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=False
    )

    model = CaptchaModel(num_chars = len(lbl_encoder.classes_))
    model.to(config.DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.8, patience=5, verbose=True
    )

    train_loss_data = []
    test_loss_data = []
    for epoch in range(config.EPOCHS):
        train_loss = engine.train_fn(model, train_loader, optimizer, save_model=True)
        eval_preds, test_loss = engine.eval_fn(model, test_loader)

        eval_captcha_preds = []
        for vp in eval_preds:
            current_preds = decode_predictions(vp, lbl_encoder)
            eval_captcha_preds.extend(current_preds)

        combined = list(zip(test_orig_targets, eval_captcha_preds))

        pprint(combined[:10])
        test_dup_rem = [remove_duplicates(c) for c in test_orig_targets]
        accuracy = metrics.accuracy_score(test_dup_rem, eval_captcha_preds)
        print(
            f"Epoch={epoch}, Train Loss={train_loss}, Test Loss={test_loss} Accuracy={accuracy}"
        )
        scheduler.step(test_loss)
        train_loss_data.append(train_loss)
        test_loss_data.append(test_loss)

    # print(train_dataset[0])
    plot_loss(train_loss_data, test_loss_data)
    print("done")


if __name__ == '__main__':
    run_training()
