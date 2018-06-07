"""
Starting point for your journey

This code is written in Python 3
(don't attempt to use Python 2 or you'll waste your time with unicode issues)
"""
import gc
import threading
import zipfile
from time import sleep

import lightgbm as lightgbm
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder

if __name__ == '__main__':

    # Read in the dataset
    with zipfile.ZipFile('data.zip', 'r').open('data.csv') as fp:
        df_data = pd.read_csv(fp).fillna('')

    tag_data = pd.read_csv('label_info.csv')
    # Create the set of level 2 tags
    l2_tags = set(tag_data[tag_data['level'] == 2]['name'])

    # Get level 2 tags for each product
    y = df_data['labels'].apply(lambda x: x.split(','))  # split the tags in each row
    y = y.apply(lambda x: set(x) & l2_tags)  # only select the tags at level 2

    # One-hot encoder for the product tags
    mlb = MultiLabelBinarizer()
    y_bin = mlb.fit_transform(y)

    dic_labels = {i: '{}'.format(c.lower()) for i, c in enumerate(mlb.classes_)}
    df_labels = pd.DataFrame(y_bin, columns=list(dic_labels))

    clusters = [[0, 22, 16, 20], [2, 5, 10, 41],
                [40, 50, 3, 13], [15, 33, 4, 47], [17, 6, 28, 56],
                [32, 14, 34], [57, 44, 45], [48, 1, 58], [12],
                [55, 42, 36, 37, 31, 59], [24, 25],
                [38, 53], [18, 27], [8, 49], [7, 23], [29, 46],
                [19],[43],[35],[30],[39],[9],[51],[21],[26],[11],[54],[52]]

    for i, cluster in enumerate(clusters):
        for j in cluster:
            if j in dic_labels:
                print('cluster: {}, id: {}, label: {}'.format(i, j, dic_labels[j]))

    def find_clusters(row):
        for k, v in dic_labels.items():
            if row[k] == 1:
                for i, cluster in enumerate(clusters):
                    if k in cluster:
                        return i

        return -1

    # Find cluster id of labels in our dataset we created by using dendrogram on heatmap in EDA notebook.
    # In other words, we applied a Hierarchical clustering by using heatmap and dendrogram on top of correlation matrix.
    df_data['cluster_id'] = df_labels.apply(find_clusters, 1).reset_index(drop=True)

    # Calculate count of cluster per cluster by aggregation
    df_agg = df_data[['cluster_id']].groupby(['cluster_id']).aggregate({'cluster_id' : ['count']}).sort_values([('cluster_id', 'count')])
    print(df_agg, len(df_data))

    # Filtering out the most frequently occurring clusters in data
    f_list = df_agg[df_agg[('cluster_id', 'count')] > 100].index.tolist()
    df_data = df_data[df_data['cluster_id'].apply(lambda x: x in f_list)]

    # Re-arranging cluster ids
    lblEncoder = LabelEncoder()
    lblEncoder.fit(df_data['cluster_id'])
    df_data['cluster_id'] = lblEncoder.transform(df_data['cluster_id'])

    # Get raw text from products
    df_data_X = df_data[['name', 'description', 'brand']].apply(lambda x: ' '.join(x), axis=1)
    df_data_y = df_data['cluster_id']
    num_class = len(np.unique(df_data_y))

    # Split the data between training and test set
    df_train_X, df_text_X, df_train_y, df_test_y = train_test_split(df_data_X, df_data_y, test_size=0.2, random_state=112358)

    def run():
        __K = 3
        __r = 3
        __N = __K * __r
        __delay = 10  # second

        __threads = []
        __predictions = pd.DataFrame(columns=list(range(1, __N + 1)), index=range(len(df_test_y)))

        nltk.download('stopwords')
        # Collecting Stopwords for each different language, Russian, German and English
        rus = stopwords.words('russian')
        de = stopwords.words('german')
        eng = stopwords.words('english')
        sws = set(rus + de + eng)

        __count_vector_params = {
            'ngram_range': (1, 2),
            'stop_words': sws,
            'max_features': 500,
            'dtype': np.float32,
            'token_pattern': r'\w{1,}'
        }

        __tfidf_params = {
            'sublinear_tf': True,
            'norm': 'l2',
            'smooth_idf': False
        }

        __svd_params = {
            'n_components': 50,
            'random_state': 12435
        }

        __lgbm_params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'multiclass',
            'metric': 'auc',
            'max_depth': 32,
            'num_leaves': 100,
            'learning_rate': 0.01,
            'verbose': 0,
            'seed': 12341,
            'colsample_bytree': 0.95,
            'subsample': 0.95,
            'num_class' : num_class,
        }

        def __run(k, train_index, valid_index):

            X_train, X_valid = df_train_X.iloc[train_index].copy().reset_index(drop=True), \
                               df_train_X.iloc[valid_index].copy().reset_index(drop=True)

            y_train, y_valid = df_train_y.iloc[train_index].copy().reset_index(drop=True), \
                               df_train_y.iloc[valid_index].copy().reset_index(drop=True)

            X_test = df_text_X.copy()

            X_train, X_valid, X_test = __transform_X(k, [X_train.values, X_valid.values, X_test.values])

            model = __create_model(k, X_train, y_train, X_valid, y_valid)
            __predictions[k] = model.predict(X_test)

            __free([X_train, X_valid, X_test, y_train, y_valid, model])

        def __transform_X(k, Xs):

            print('Fold#{} Transformation was started'.format(k))
            assert len(Xs) == 3

            pipeline = Pipeline([
                ('bow', CountVectorizer(**__count_vector_params)),
                ('tfidf', TfidfTransformer(**__tfidf_params)),
                ('svd', TruncatedSVD(**__svd_params)),
            ])

            pipeline.fit(Xs[0])
            result =  tuple([pipeline.transform(x) for x in Xs])
            print('Fold#{} Transformation was done'.format(k))

            return result

        def __create_model(k, X_train, y_train, X_valid, y_valid):

            print('Fold#{} Model creation was started'.format(k))

            train_dataset = lightgbm.Dataset(X_train,
                                             y_train)

            valid_dataset = lightgbm.Dataset(X_valid,
                                             y_valid)

            model = lightgbm.train(
                __lgbm_params,
                train_dataset,
                num_boost_round=500,
                valid_sets=[train_dataset, valid_dataset],
                valid_names=['train-{}-{}'.format(k, __N), 'valid-{}-{}'.format(k, __N)],
                early_stopping_rounds=50,
                verbose_eval=5,
            )

            print('Fold#{} Model creation was done'.format(k))

            return model

        def __free(objs):
            for x in objs:
                del x
            gc.collect()

        def __evaluate():
            __wait()
            y_test_prediction = __predictions.mean(1)

            auc_scores = {}
            pre_scores = {}
            fpr = {}
            tpr = {}

            pre_micro = {}
            pre_macro = {}

            for c in np.unique(df_test_y):
                y_test_actual = df_test_y.copy()
                y_test_actual = (y_test_actual == c) * 1

                fpr[c], tpr[c], _ = roc_curve(y_test_actual, y_test_prediction)
                auc_scores[c] = roc_auc_score(y_test_actual, y_test_prediction)
                pre_scores[c] =  fpr[c] / (fpr[c] + tpr[c])

                # https: // sebastianraschka.com / faq / docs / multiclass - metric.html
                pre_micro[c] = sum(fpr[c]) / sum(fpr[c] + tpr[c])
                pre_macro[c] = np.mean(pre_scores[c])

            df_res = pd.DataFrame()
            df_res['cluster_id'] = lblEncoder.inverse_transform(np.unique(df_test_y))
            df_res['labels'] = df_res['cluster_id'].apply(lambda c: [dic_labels[k] for k in clusters[c]])
            df_res['auc'] = list(auc_scores.values())
            df_res['pre_micro'] = list(pre_micro.values())
            df_res['pre_macro'] = list(pre_macro.values())

            df_des = df_res.describe()

            df_res.to_csv('result.csv', index=False)
            df_des.to_csv('statistics.csv')


        def __wait():
            for k in range(__N):
                __threads[k].join()
                print('Fold#{} was done...'.format(k))

        kf = RepeatedStratifiedKFold(n_splits=__K, n_repeats=__r, random_state=112358)

        k = 1
        for train_index, valid_index in kf.split(df_train_X, df_train_y):
            t = threading.Thread(target=__run, args=(k, train_index, valid_index))

            t.start()
            __threads.append(t)

            print('Sleeping {} seconds'.format(__delay))
            sleep(__delay)

            print('Fold#{} was started...'.format(k))
            k += 1

        __evaluate()

    run()