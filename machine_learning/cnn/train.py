import os
import cv2
from verifier import create_model, preprocess_input
from sklearn.metrics import classification_report
import numpy as np
import tqdm
import glob

def load_image_pairs(path):
    files = glob.glob(path+'/**/*.*')
    x1, x2 = [], []
    y = []
    for master in tqdm.tqdm(files, 'Loading data...'):
        if not master.split('.')[-2].endswith('__m'):
            continue
        rend_name = master.replace('__m','__r')
        if not os.path.exists(rend_name):
            continue
        # append RGB image
        x1.append(cv2.imread(master)[..., ::-1])
        x2.append(cv2.imread(rend_name)[..., ::-1])
        y.append([0, 1] if master.split(os.sep)[-2]=='tamper' else [1, 0])
    x1 = preprocess_input(np.array(x1, dtype=np.uint8))
    x2 = preprocess_input(np.array(x2, dtype=np.uint8))
    y = np.array(y, dtype=np.uint8)
    return x1, x2, y

if __name__ == '__main__':
    path = '../../../data/cnn/'
    x1, x2, y = load_image_pairs(path)
    np.random.seed(1337)
    test_idx = np.random.choice(x1.shape[0], int(0.2*x1.shape[0]), False)
    train_idx = list(set(range(x1.shape[0])).difference(test_idx))
    x1_test, x2_test, y_test = x1[test_idx], x2[test_idx], y[test_idx]
    x1_train, x2_train, y_train = x1[train_idx], x2[train_idx], y[train_idx]
    model = create_model()
    history = model.fit(
        [x1_train, x2_train],
        y_train,
        batch_size=512,
        epochs=300,
        shuffle=True
    )
    y_pred = model.predict([x1_test, x2_test])
    model.save('../output/verifier_cnn.hdf5')
    y_pred_label = y_pred[..., 1] > 0.5
    y_test_label = y_test[..., 1] > 0.5
    print(classification_report(y_test_label, y_pred_label))

