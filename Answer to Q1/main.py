# -*- coding:utf-8 -*-
import tensorflow as tf
import matplotlib.pyplot as plt
import util


if __name__ == '__main__':
    # induce MNIST data set
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # implement a simpler version of the method using the MNIST data set for regression on digit 0 and digit 7
    # package the raw data into bags and set tags
    # deal with train data
    x_train_toy, y_train_toy = util.setDataBag(y_train, x_train)

    # deal with test data
    x_test_toy, y_test_toy = util.setDataBag(y_train, x_train)

    # set the model of machine learning
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(100, 28, 28)),
        # L2 regularization
        tf.keras.layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        # fight overwriting
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128),
        tf.keras.layers.Dense(100),
    ])

    # set parameters as in the paper
    initial_learning_rate = 0.0001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=100, decay_rate=0.0005, staircase=True
    )
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule)
    model.compile(
        # adam optimizer
        optimizer='adam',
        # loss function
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    # train
    model.fit(x_train_toy, y_train_toy, epochs=20, batch_size=1)
    probability_model = tf.keras.Sequential([model,
                                             tf.keras.layers.Softmax()])

    # predict
    predictions = probability_model.predict(x_test_toy)

    # show the predict accuracy
    test_loss, test_acc = model.evaluate(x_test_toy, y_test_toy, verbose=2)

    print('\nTest accuracy:', test_acc)

    # visual(set bag 1 as example)
    i = 10
    plt.figure(figsize=(23, 10))
    plt.subplot(211)
    util.plot_image(i, predictions[i], y_test_toy, x_test_toy)
    plt.subplot(212)
    util.plot_value_array(i, predictions[i], y_test_toy)
    plt.tight_layout()
    plt.show()

