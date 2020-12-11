import os
import glob
import torch
import numpy as np

from sklearn import preprocessing
from sklearn import model_selection
from sklearn import metrics

import config
import dataset


def run_training():
    image_files = glob.glob(os.path.join(config.DATA_DIR, "*.png"))
    # "../xywz.png" -> "xywz"
    targets_orig = [x.split("/")[-1].split["."][0] for x in image_files]
    # separate the targets on character level
    targets = [[char for char in x] for x in targets_orig]
    targets_flat = [c for clist in targets for c in clist]

    lbl_encoder = preprocessing.LabelEncoder()
    lbl_encoder.fit(targets_flat)

    print("done")


if __name__ == '__main__':
    run_training()