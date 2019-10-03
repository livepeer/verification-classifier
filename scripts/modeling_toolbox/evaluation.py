import numpy as np
from sklearn.metrics import fbeta_score, roc_curve, auc, confusion_matrix
from sklearn.decomposition import PCA
from sklearn import random_projection
from sklearn import svm
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from keras.layers import Dense, Input, Dropout
from keras.models import Model
from keras import regularizers
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l2
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb


def one_class_svm(x_train, x_test, x_attacks, svm_results):

    # SVM Hyper-parameters
    nus = [0.01]
    gammas = ['auto']
    dimensions = [int(i*x_test.shape[1]) for i in [0.25, 0.35, 0.5, 0.75, 0.9, 1]]
    dimensions = list(filter(lambda x: x > 0, dimensions))

    for n in dimensions:

        x_reduced_pca, test_reduced_pca, attack_reduced_pca = reduce_dimensionality(n, x_train, x_test, 'PCA',
                                                                                    attack=x_attacks)
  
        for nu in nus:
            for gamma in gammas:

                # Fit classifier with PCA reduced data
                classifier = svm.OneClassSVM(kernel='rbf', gamma=gamma, nu=nu, cache_size=7000)
                classifier.fit(x_reduced_pca)
                fb, area, tnr, tpr_train, tpr_test = unsupervised_evaluation(classifier, x_reduced_pca,
                                                                             test_reduced_pca,
                                                                             attack_reduced_pca)

                svm_results = svm_results.append({'nu': nu, 'gamma': gamma, 'n_components': n, 'TPR_train': tpr_train,
                                                  'TPR_test': tpr_test, 'TNR': tnr, 'model': 'svm', 'auc': area,
                                                  'f_beta': fb, 'projection': 'PCA'}, ignore_index=True)

                # Fit classifier with RP reduced data
                classifier = svm.OneClassSVM(kernel='rbf', gamma=gamma, nu=nu, cache_size=7000)

  
                classifier.fit(x_train)
                fb, area, tnr, tpr_train, tpr_test = unsupervised_evaluation(classifier, x_train,
                                                                             x_test, x_attacks)

                svm_results = svm_results.append({'nu': nu, 'gamma': gamma, 'n_components': x_test.shape[1],
                                                  'TPR_train': tpr_train,
                                                  'TPR_test': tpr_test, 'TNR': tnr, 'model': 'svm', 'auc': area,
                                                  'f_beta': fb, 'projection': 'None'}, ignore_index=True)

    return svm_results


def isolation_forest(x_train, x_test, x_attacks, isolation_results):

    # Isolation Forest Hyper-parameters
    estimators = [200, 100]
    contaminations = [0.01]
    dimensions = [int(i*x_test.shape[1]) for i in [0.25, 0.5, 0.9, 1]]
    dimensions = list(filter(lambda x: x > 0, dimensions))

    for n in dimensions:

        x_reduced_pca, test_reduced_pca, attack_reduced_pca = reduce_dimensionality(n, x_train, x_test, 'PCA',
                                                                                    attack=x_attacks)
        x_reduced_rp, test_reduced_rp, attack_reduced_rp = reduce_dimensionality(n, x_train, x_test, 'RP',
                                                                                 attack=x_attacks)

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


def neural_network(x_train, y_train, x_test, y_test):
    model = Sequential()

    model.add(Dense(128, input_shape=(x_train.shape[1],), activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.1))

    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.2))

    model.add(Dense(128, kernel_initializer='glorot_uniform', activation='sigmoid'))
    model.add(Dropout(0.4))

    model.add(Dense(64, kernel_initializer='glorot_uniform', activation='tanh'))
    model.add(Dropout(0.5))

    model.add(Dense(32, kernel_initializer='glorot_uniform', activation='tanh'))
    model.add(Dropout(0.4))

    model.add(Dense(128, kernel_initializer='glorot_uniform', activation='tanh'))
    model.add(Dropout(0.3))

    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    network_history = model.fit(x_train, y_train, batch_size=128, epochs=10, verbose=0,
                                validation_data=(x_test, y_test))
    plot_history_with_acc(network_history)
    return model


def random_forest(x_train, y_train, x_test, y_test, random_forest_results):

    # Random forest Hyper-parameters
    estimators = [150, 200]
    dimensions = [int(i*x_test.shape[1]) for i in [1]]

    for estimator in estimators:

        for n in dimensions:

            x_reduced_pca, test_reduced_pca = reduce_dimensionality(n, x_train, x_test, 'PCA')
            x_reduced_rp, test_reduced_rp = reduce_dimensionality(n, x_train, x_test, 'RP')

            classifier = RandomForestClassifier(n_estimators=estimator, n_jobs=7)

            classifier.fit(x_reduced_pca, y_train)
            fb, area, tnr, tpr = supervised_evaluation(classifier, test_reduced_pca, y_test)

            random_forest_results = random_forest_results.append({'estimators': estimator,
                                                                  'n_components': n,
                                                                  'TPR': tpr,
                                                                  'TNR': tnr,
                                                                  'model': 'random_forest',
                                                                  'auc': area,
                                                                  'f_beta': fb,
                                                                  'projection': 'PCA'}, ignore_index=True)

            classifier = RandomForestClassifier(n_estimators=estimator, n_jobs=7)

            classifier.fit(x_reduced_rp, y_train)
            fb, area, tnr, tpr = supervised_evaluation(classifier, test_reduced_rp, y_test)

            random_forest_results = random_forest_results.append({'estimators': estimator,
                                                                  'n_components': n,
                                                                  'TPR': tpr,
                                                                  'TNR': tnr,
                                                                  'model': 'random_forest',
                                                                  'auc': area,
                                                                  'f_beta': fb,
                                                                  'projection': 'RP'}, ignore_index=True)

            classifier = RandomForestClassifier(n_estimators=estimator, n_jobs=7)

        classifier.fit(x_train, y_train)
        fb, area, tnr, tpr = supervised_evaluation(classifier, x_test, y_test)

        random_forest_results = random_forest_results.append({'estimators': estimator,
                                                              'n_components': x_test.shape[1],
                                                              'TPR': tpr,
                                                              'TNR': tnr,
                                                              'model': 'random_forest',
                                                              'auc': area,
                                                              'f_beta': fb,
                                                              'projection': 'None'}, ignore_index=True)

    return random_forest_results


def ada_boost(x_train, y_train, x_test, y_test, ada_boost_results):

    # AdaBoost Hyper-parameters
    learning_rates = [0.55]
    dimensions = [int(i*x_test.shape[1]) for i in [1]]

    for n in dimensions:

        x_reduced_pca, test_reduced_pca = reduce_dimensionality(n, x_train, x_test, 'PCA')
        x_reduced_rp, test_reduced_rp = reduce_dimensionality(n, x_train, x_test, 'RP')

        for lr in learning_rates:

            classifier = AdaBoostClassifier(learning_rate=lr)

            classifier.fit(x_reduced_pca, y_train)
            fb, area, tnr, tpr = supervised_evaluation(classifier, test_reduced_pca, y_test)

            ada_boost_results = ada_boost_results.append({'LR': lr,
                                                          'n_components': n,
                                                          'TPR': tpr,
                                                          'TNR': tnr,
                                                          'model': 'ada_boost',
                                                          'auc': area,
                                                          'f_beta': fb,
                                                          'projection': 'PCA'}, ignore_index=True)

            classifier = AdaBoostClassifier(learning_rate=lr)

            classifier.fit(x_reduced_rp, y_train)
            fb, area, tnr, tpr = supervised_evaluation(classifier, test_reduced_rp, y_test)

            ada_boost_results = ada_boost_results.append({'LR': lr,
                                                          'n_components': n,
                                                          'TPR': tpr,
                                                          'TNR': tnr,
                                                          'model': 'ada_boost',
                                                          'auc': area,
                                                          'f_beta': fb,
                                                          'projection': 'RP'}, ignore_index=True)
    return ada_boost_results


def svm_classifier(x_train, y_train, x_test, y_test, svm_results):

    # SVC Hyper-parameters
    dimensions = [int(i*x_test.shape[1]) for i in [1]]

    for n in dimensions:

        x_reduced_pca, test_reduced_pca = reduce_dimensionality(n, x_train, x_test, 'PCA')
        x_reduced_rp, test_reduced_rp = reduce_dimensionality(n, x_train, x_test, 'RP')

        classifier = svm.SVC(gamma='auto', cache_size=7000)

        classifier.fit(x_reduced_pca, y_train)
        fb, area, tnr, tpr = supervised_evaluation(classifier, test_reduced_pca, y_test)

        svm_results = svm_results.append({
                                          'n_components': n,
                                          'TPR': tpr,
                                          'TNR': tnr,
                                          'model': 'svm',
                                          'auc': area,
                                          'f_beta': fb,
                                          'projection': 'PCA'}, ignore_index=True)

        classifier = svm.SVC(gamma='auto', cache_size=7000)

        classifier.fit(x_reduced_rp, y_train)
        fb, area, tnr, tpr = supervised_evaluation(classifier, test_reduced_rp, y_test)

        svm_results = svm_results.append({
                                          'n_components': n,
                                          'TPR': tpr,
                                          'TNR': tnr,
                                          'model': 'svm',
                                          'auc': area,
                                          'f_beta': fb,
                                          'projection': 'RP'}, ignore_index=True)
    return svm_results


def xg_boost(x_train, y_train, x_test, y_test, xg_boost_results):

    # XGBoost Hyper-parameters
    dimensions = [int(i*x_test.shape[1]) for i in [1]]

    for n in dimensions:

        x_reduced_pca, test_reduced_pca = reduce_dimensionality(n, x_train, x_test, 'PCA')
        x_reduced_rp, test_reduced_rp = reduce_dimensionality(n, x_train, x_test, 'RP')

        classifier = xgb.XGBClassifier()
        grid = {'max_depth': 10}
        classifier.set_params(**grid)

        classifier.fit(x_reduced_pca, y_train)
        fb, area, tnr, tpr = supervised_evaluation(classifier, test_reduced_pca, y_test)

        xg_boost_results = xg_boost_results.append({
            'n_components': n,
            'TPR': tpr,
            'TNR': tnr,
            'model': 'xgboost',
            'auc': area,
            'f_beta': fb,
            'projection': 'PCA'}, ignore_index=True)

        classifier = xgb.XGBClassifier()
        grid = {'max_depth': 10}
        classifier.set_params(**grid)

        classifier.fit(x_reduced_rp, y_train)
        fb, area, tnr, tpr = supervised_evaluation(classifier, test_reduced_rp, y_test)

        xg_boost_results = xg_boost_results.append({
            'n_components': n,
            'TPR': tpr,
            'TNR': tnr,
            'model': 'xgboost',
            'auc': area,
            'f_beta': fb,
            'projection': 'RP'}, ignore_index=True)

    classifier = xgb.XGBClassifier()
    grid = {'max_depth': 10}
    classifier.set_params(**grid)

    classifier.fit(x_train, y_train)
    fb, area, tnr, tpr = supervised_evaluation(classifier, x_test, y_test)

    xg_boost_results = xg_boost_results.append({
        'n_components': x_test.shape[1],
        'TPR': tpr,
        'TNR': tnr,
        'model': 'xgboost',
        'auc': area,
        'f_beta': fb,
        'projection': 'None'}, ignore_index=True)

    return xg_boost_results


def supervised_evaluation(classifier, x_test, y_test, beta=20, nn=False):

    if not nn:
        y_pred = classifier.predict(x_test)
        confusion_matrix(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        fb = fbeta_score(y_test, y_pred, beta=beta, pos_label=1)
        area = auc(fpr, tpr)
        tpr = tpr[1]
        tnr = 1 - fpr[1]
        return fb, area, tnr, tpr


def plot_roc(classifier, test, attacks, title):

    y_pred_test = classifier.predict(test)
    y_pred_outliers = classifier.predict(attacks)

    fpr, tpr, _ = roc_curve(np.concatenate([np.ones(y_pred_test.shape[0]),
                                            -1*np.ones(y_pred_outliers.shape[0])]),
                            np.concatenate([y_pred_test, y_pred_outliers]), pos_label=1)
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
    plt.legend(loc='lower right')
    plt.show()


def plot_roc_supervised(classifier, x_test, y_test, title, nn=False):

    y_pred = classifier.predict(x_test)
    fpr, tpr, _ = roc_curve(y_test, y_pred)

    if nn:
        y_pred = [round(x[0]) for x in y_pred]
    print(confusion_matrix(y_test, y_pred))

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
    plt.title('Receiver operating characteristic {}'.format(title))
    plt.legend(loc='lower right')
    plt.show()


def plot_history(network_history, title):
    plt.figure(figsize=(10, 5))
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.semilogy(network_history.history['loss'])
    plt.semilogy(network_history.history['val_loss'])
    plt.legend(['Training', 'Validation'])
    plt.show()


def plot_history_with_acc(network_history, title='Loss and Accuracy'):
    plt.figure(figsize=(15, 10))
    plt.subplot(211)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.semilogy(network_history.history['loss'])
    plt.semilogy(network_history.history['val_loss'])
    plt.legend(['Training', 'Validation'])

    plt.subplot(212)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(network_history.history['acc'])
    plt.plot(network_history.history['val_acc'])
    plt.legend(['Training', 'Validation'], loc='lower right')
    plt.show()


def reduce_dimensionality(n_components, train, test, method, attack=None):
    if method == 'PCA':
        matrix = PCA(n_components=n_components)
    elif method == 'RP':
        matrix = random_projection.SparseRandomProjection(n_components=n_components, random_state=7)
    else:
        print('unknown projection method, choose either RP or PCA')
        return None

    train = matrix.fit_transform(train)
    test = matrix.transform(test)

    if attack is None:
        return train, test

    attack = matrix.transform(attack)
    return train, test, attack
