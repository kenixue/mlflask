import dataset
import tensorflow as tf
import numpy as np
import csv

TRAINING_SET_FRACTION = 0.95


def main(argv, serum_sodium=None):
    data = dataset.Dataset('data/heart.csv')
    print(data.processed_results)


    train_results_len = int(TRAINING_SET_FRACTION * len(data.processed_results))
    train_results = data.processed_results[:train_results_len]
    test_results = data.processed_results[train_results_len:]

    def map_results(results):
        features = {}

        for result in results:
            for key in result.keys():
                if key not in features:
                    features[key] = []
                features[key].append(result[key])

        for key in features.keys():
            features[key] = np.array(features[key])

        re = features['DEATH_EVENT']
        del features['DEATH_EVENT']

        return features, re

    train_features, train_labels = map_results(train_results)

    test_features, test_labels = map_results(test_results)
    # 修改
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=train_features,
        y=train_labels,
        batch_size=500,
        num_epochs=None,
        shuffle=True
    )

    def train_input_fn():
        features = train_features
        labels = train_labels
        return features, labels

    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=test_features,
        y=test_labels,
        num_epochs=1,
        shuffle=False
    )


    feature_columns = []
    feature_columns = feature_columns + [
        tf.feature_column.numeric_column(key='age'),
        tf.feature_column.numeric_column(key='anaemia'),
        tf.feature_column.numeric_column(key='creatinine_phosphokinase'),
        tf.feature_column.numeric_column(key='diabetes'),
        tf.feature_column.numeric_column(key='ejection_fraction'),
        tf.feature_column.numeric_column(key='high_blood_pressure'),
        tf.feature_column.numeric_column(key='platelets'),
        tf.feature_column.numeric_column(key='serum_creatinine'),
        tf.feature_column.numeric_column(key='serum_sodium'),
        tf.feature_column.numeric_column(key='sex'),
        tf.feature_column.numeric_column(key='smoking'),
        tf.feature_column.numeric_column(key='time'),
    ]



    model = tf.estimator.DNNClassifier(
        model_dir='model/',
        hidden_units=[10,10],
        feature_columns=feature_columns,
        n_classes=2,
        label_vocabulary=['0', '1'],
        optimizer=tf.train.ProximalAdagradOptimizer(
            learning_rate=0.1,
            l1_regularization_strength=0.001
        ))

    with open('training-log.csv', 'w') as stream:
        csvwriter = csv.writer(stream)
        step = 10
        for i in range(0, 10):
            model.train(input_fn=train_input_fn, steps=step)
            evaluation_result = model.evaluate(input_fn=test_input_fn)

            # predictions = list(model.predict(input_fn=test_input_fn))

            csvwriter.writerow([(i + 1) * step, evaluation_result['accuracy'], evaluation_result['average_loss']])


if __name__ == '__main__':
    #tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
