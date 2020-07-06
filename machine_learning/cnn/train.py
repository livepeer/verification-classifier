import os
import cv2
from verifier import *
from sklearn.metrics import classification_report
import numpy as np
import tqdm
import glob
import random
import tensorflow as tf

def load_image_pairs(path):
    files = list(glob.glob(path + '/**/*.*'))
    random.shuffle(files)
    files = files[:40000]
    x1, x2 = [], []
    y = []
    names = []
    i = 0
    for master in tqdm.tqdm(files, 'Loading data...'):
        i += 1
        if not master.split('.')[-2].endswith('__m'):
            continue
        if 'bitrate' in master:
            continue
        rend_name = master.replace('__m', '__r')
        if not os.path.exists(rend_name):
            continue
        # append RGB image
        master_img = cv2.imread(master)[..., ::-1]
        rend_img = cv2.imread(rend_name)[..., ::-1]
        if master_img.shape[:2] != IMG_SHAPE[:2]:
            master_img = cv2.resize(master_img, IMG_SHAPE[:2])
        if rend_img.shape[:2] != IMG_SHAPE[:2]:
            rend_img = cv2.resize(rend_img, IMG_SHAPE[:2])
        x1.append(master_img)
        x2.append(rend_img)
        names.append(master)
        y.append([0, 1] if master.split(os.sep)[-2] == 'tamper' else [1, 0])
    x1 = preprocess_input(np.array(x1, dtype=np.uint8))
    x2 = preprocess_input(np.array(x2, dtype=np.uint8))
    y = np.array(y, dtype=np.uint8)
    return np.stack([x1, x2], axis=4), y, names


if __name__ == '__main__':
    path = '/win2/data/livepeer/images/'
    x_path, y_path = '/win2/data/livepeer/images/x.npz', '/win2/data/livepeer/images/y.npz'
    random.seed(1337)
    np.random.seed(1337)

    if not os.path.exists(x_path) or not os.path.exists(y_path):
        x, y, names = load_image_pairs(path)
        # np.savez(x_path, x)
        # np.savez(y_path, y)
    else:
        x = np.load(x_path)
        x = x.get(list(x.keys())[0])
        y = np.load(y_path)
        y = y.get(list(y.keys())[0])

    test_fraction = 0.2

    test_idx = np.random.choice(x.shape[0], int(0.2 * x.shape[0]), False)
    train_idx = list(set(range(x.shape[0])).difference(test_idx))

    x_test, y_test = x[test_idx], y[test_idx]
    x_train, y_train = x[train_idx], y[train_idx]

    model = create_model()

    checkpoint_filepath = '../../../data/checkpoint'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='loss',
        mode='min',
        save_best_only=True)

    history = model.fit(
        [x_train[..., 0], x_train[..., 1]],
        y_train,
        batch_size=64,
        epochs=100,
        shuffle=True,
        callbacks = [model_checkpoint_callback]
    )
    y_pred = model.predict([x_test[..., 0], x_test[..., 1]])
    model.save('../output/verifier_cnn.hdf5')
    y_pred_label = y_pred[..., 1] > 0.5
    y_test_label = y_test[..., 1] > 0.5
    print(classification_report(y_test_label, y_pred_label))
