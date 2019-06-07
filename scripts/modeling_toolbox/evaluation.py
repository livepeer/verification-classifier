import numpy as np
from sklearn.metrics import fbeta_score, roc_curve, auc
from sklearn.decomposition import PCA
from sklearn import random_projection
from sklearn import svm
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from keras.layers import Dense, Input
from keras.optimizers import Adam
from keras.models import Model
from keras import regularizers


def one_class_svm(x_train, x_test, x_attacks, svm_results):

    # SVM Hyper-parameters
    nus = [0.01]
    gammas = ['auto']
    dimensions = [int(i*x_test.shape[1]) for i in [0.25, 0.35, 0.5, 0.75, 0.9, 1]]

    for n in dimensions:

        # Use PCA for dim reduction
        pca = PCA(n_components=n)
        x_reduced_pca = pca.fit_transform(x_train)
        test_reduced_pca = pca.transform(x_test)
        attack_reduced_pca = pca.transform(x_attacks)

        # Use Random Projections for dim reduction
        rp = random_projection.SparseRandomProjection(n_components=n)
        x_reduced_rp = rp.fit_transform(x_train)
        test_reduced_rp = rp.transform(x_test)
        attack_reduced_rp = rp.transform(x_attacks)

        for nu in nus:
            for gamma in gammas:

                # Fit classifier with PCA reduced data
                classifier = svm.OneClassSVM(kernel='rbf',gamma=gamma, nu=nu, cache_size=7000)
                classifier.fit(x_reduced_pca)
                fb, area, tnr, tpr_train, tpr_test = unsupervised_evaluation(classifier, x_reduced_pca,
                                                                             test_reduced_pca,
                                                                             attack_reduced_pca)

                svm_results = svm_results.append({'nu': nu, 'gamma': gamma, 'n_components': n, 'TPR_train': tpr_train,
                                                  'TPR_test': tpr_test, 'TNR': tnr,'model': 'svm', 'auc': area,
                                                  'f_beta': fb, 'projection': 'PCA'}, ignore_index=True)

                # Fit classifier with RP reduced data
                classifier = svm.OneClassSVM(kernel='rbf',gamma=gamma, nu=nu, cache_size=7000)

                classifier.fit(x_reduced_rp)
                fb, area, tnr, tpr_train, tpr_test = unsupervised_evaluation(classifier, x_reduced_rp,
                                                                             test_reduced_rp, attack_reduced_rp)

                svm_results = svm_results.append({'nu': nu, 'gamma': gamma, 'n_components': n, 'TPR_train': tpr_train,
                                                  'TPR_test': tpr_test, 'TNR': tnr, 'model': 'svm', 'auc': area,
                                                  'f_beta': fb, 'projection': 'RP'}, ignore_index=True)

    return svm_results


def isolation_forest(x_train, x_test, x_attacks, isolation_results):
    estimators = [200, 100]
    contaminations = [0.01]
    dimensions = [int(i*x_test.shape[1]) for i in [0.25, 0.5, 0.9, 1]]

    for n in dimensions:
        # Use PCA for dim reduction
        pca = PCA(n_components=n)
        x_reduced_pca = pca.fit_transform(x_train)
        test_reduced_pca = pca.transform(x_test)
        attack_reduced_pca = pca.transform(x_attacks)

        # Use Random Projections for dim reduction
        rp = random_projection.SparseRandomProjection(n_components=n)
        x_reduced_rp = rp.fit_transform(x_train)
        test_reduced_rp = rp.transform(x_test)
        attack_reduced_rp = rp.transform(x_attacks)
        max_features = list(range(1, n + 1, 4))
        for estimator in estimators:
            for contamination in contaminations:
                for max_feature in max_features:
                    classifier = IsolationForest(n_estimators=estimator,
                                                 contamination=contamination,
                                                 max_features=max_feature,
                                                 n_jobs=7)

                    classifier.fit(x_reduced_pca)
                    fb, area, tnr, tpr_train, tpr_test = unsupervised_evaluation(classifier, x_reduced_pca,
                                                                                 test_reduced_pca, attack_reduced_pca)

                    isolation_results = isolation_results.append({'estimators': estimator, 'contamination': contamination,
                                                                  'n_components': n, 'max_features': max_feature,
                                                                  'TPR_train': tpr_train,
                                                                  'TPR_test': tpr_test,
                                                                  'TNR': tnr,
                                                                  'model': 'isolation_forest',
                                                                  'auc': area,
                                                                  'f_beta': fb,
                                                                  'projection': 'PCA'}, ignore_index=True)

                    classifier = IsolationForest(n_estimators=estimator,
                                                 contamination=contamination,
                                                 max_features=max_feature,
                                                 n_jobs=7)
                    classifier.fit(x_reduced_rp)
                    fb, area, tnr, tpr_train, tpr_test = unsupervised_evaluation(classifier, x_reduced_rp,
                                                                                 test_reduced_rp, attack_reduced_rp)

                    isolation_results = isolation_results.append({'estimators': estimator, 'contamination': contamination,
                                                                  'n_components': n, 'max_features': max_feature,
                                                                  'TPR_train': tpr_train,
                                                                  'TPR_test': tpr_test,
                                                                  'TNR': tnr,
                                                                  'model': 'isolation_forest',
                                                                  'auc': area,
                                                                  'f_beta': fb,
                                                                  'projection': 'RP'}, ignore_index=True)
    return isolation_results


def autoencoder(x_train, x_test, x_attacks, ae_svm_results):
    latent_dim = 3
    input_vector = Input(shape=(x_train.shape[1],))
    encoded = Dense(latent_dim, activation='relu')(input_vector)
    decoded = Dense(x_train.shape[1], activity_regularizer=regularizers.l1(10e-5))(encoded)
    autoencoder = Model(input_vector, decoded)
    encoder = Model(input_vector, encoded)
    autoencoder.compile(optimizer=Adam(lr=0.001), loss='mse')
    network_history = autoencoder.fit(x_train, x_train, shuffle=True, batch_size=16, epochs=10,
                                      validation_data=(x_test, x_test), verbose=True)
    plot_history(network_history, 'AE history')
    print('Mean loss on train: {}'.format(autoencoder.evaluate(x_train, x_train, batch_size=8, verbose=False)))
    print('Mean loss on test: {}'.format(autoencoder.evaluate(x_test, x_test, batch_size=8, verbose=False)))
    print('Mean loss on attacks: {}'.format(autoencoder.evaluate(x_attacks, x_attacks, batch_size=8, verbose=False)))
    x_train_red = encoder.predict(x_train, batch_size=8)
    x_test_red = encoder.predict(x_test, batch_size=8)
    x_attacks_red = encoder.predict(x_attacks, batch_size=8)

    nus = [0.01]

    gammas = [x_train_red.shape[1], 2*x_train_red.shape[1], x_train_red.shape[1]/2, 'auto']
    for nu in nus:
        for gamma in gammas:
            classifier = svm.OneClassSVM(kernel='rbf', gamma=gamma, nu=nu, cache_size=7000)
            classifier.fit(x_train_red)

            fb, area, tnr, tpr_train, tpr_test = unsupervised_evaluation(classifier, x_train_red,
                                                                         x_test_red, x_attacks_red)

            ae_svm_results = ae_svm_results.append({'nu': nu, 'gamma': gamma, 'n_components': latent_dim,
                                                    'TPR_train': tpr_train, 'TPR_test': tpr_test, 'TNR': tnr,
                                                    'model': 'ae-svm', 'auc': area, 'f_beta': fb}, ignore_index=True)

    return ae_svm_results


def unsupervised_evaluation(classifier, train_set, test_set, attack_set, beta=20):

    y_pred_train = classifier.predict(train_set)
    y_pred_test = classifier.predict(test_set)
    y_pred_outliers = classifier.predict(attack_set)
    n_accurate_train = y_pred_train[y_pred_train == 1].size
    n_accurate_test = y_pred_test[y_pred_test == 1].size
    n_accurate_outliers = y_pred_outliers[y_pred_outliers == -1].size

    fpr, tpr, _ = roc_curve(np.concatenate([np.ones(y_pred_test.shape[0]), -1*np.ones(y_pred_outliers.shape[0])]),
                            np.concatenate([y_pred_test, y_pred_outliers]), pos_label=1)
    fb = fbeta_score(np.concatenate([np.ones(y_pred_test.shape[0]), -1*np.ones(y_pred_outliers.shape[0])]),
                     np.concatenate([y_pred_test, y_pred_outliers]), beta=beta, pos_label=1)

    tnr = n_accurate_outliers/attack_set.shape[0]
    tpr_test = n_accurate_test/test_set.shape[0]
    tpr_train = n_accurate_train/train_set.shape[0]

    area = auc(fpr, tpr)
    return fb, area, tnr, tpr_train, tpr_test


def plot_roc(classifier, test, attacks, title):
    y_pred_test = classifier.predict(test)
    y_pred_outliers = classifier.predict(attacks)

    fpr, tpr, _ = roc_curve(np.concatenate([np.ones(y_pred_test.shape[0]),
                                            -1*np.ones(y_pred_outliers.shape[0])]),
                            np.concatenate([y_pred_test, y_pred_outliers]) , pos_label=1)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic: {}'.format(title))
    plt.legend(loc="lower right")
    plt.show()


def plot_history(network_history, title):
    plt.figure(figsize=(10, 5))
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(network_history.history['loss'])
    plt.plot(network_history.history['val_loss'])
    plt.legend(['Training', 'Validation'])
    plt.show()


