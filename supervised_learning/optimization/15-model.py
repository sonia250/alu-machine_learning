# First, let me create a simpler version without f-strings

cat > supervised_learning/optimization/15-model.py << 'EOF'
#!/usr/bin/env python3
"""Module for complete neural network model with all optimizations"""
import tensorflow as tf
import numpy as np


def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9,
          beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32, epochs=5,
          save_path='/tmp/model.ckpt'):
    """
    Builds, trains, and saves a neural network using Adam, mini-batch,
    learning rate decay, and batch normalization

    Args:
        Data_train: tuple of (X_train, Y_train)
        Data_valid: tuple of (X_valid, Y_valid)
        layers: list of nodes in each layer
        activations: list of activation functions
        alpha: learning rate
        beta1: weight for first moment of Adam
        beta2: weight for second moment of Adam
        epsilon: small number to avoid division by zero
        decay_rate: decay rate for learning rate
        batch_size: mini-batch size
        epochs: number of epochs
        save_path: where to save model

    Returns:
        the path where model was saved
    """
    X_train, Y_train = Data_train
    X_valid, Y_valid = Data_valid

    x = tf.placeholder(tf.float32, shape=[None, X_train.shape[1]], name='x')
    y = tf.placeholder(tf.float32, shape=[None, Y_train.shape[1]], name='y')

    is_training = tf.placeholder(tf.bool, name='is_training')
    pred = x
    for i, (layer_size, activation) in enumerate(zip(layers, activations)):
        with tf.variable_scope('layer_{}'.format(i + 1)):
            initializer = tf.contrib.layers.variance_scaling_initializer(
                mode="FAN_AVG"
            )
            pred = tf.layers.dense(
                pred, layer_size, activation=None,
                kernel_initializer=initializer
            )
            pred = tf.layers.batch_normalization(
                pred, training=is_training, epsilon=1e-8
            )
            if activation is not None:
                pred = activation(pred)

    loss = tf.losses.softmax_cross_entropy(y, pred)
    correct = tf.equal(tf.argmax(y, 1), tf.argmax(pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    global_step = tf.Variable(0, trainable=False)
    decayed_lr = tf.train.inverse_time_decay(
        alpha, global_step, 1, decay_rate, staircase=True
    )

    optimizer = tf.train.AdamOptimizer(
        decayed_lr, beta1=beta1, beta2=beta2, epsilon=epsilon
    )
    train_op = optimizer.minimize(loss, global_step=global_step)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(epochs + 1):
            train_cost, train_acc = sess.run(
                [loss, accuracy],
                feed_dict={x: X_train, y: Y_train, is_training: False}
            )
            valid_cost, valid_acc = sess.run(
                [loss, accuracy],
                feed_dict={x: X_valid, y: Y_valid, is_training: False}
            )

            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_acc))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_acc))

            if epoch < epochs:
                perm = np.random.permutation(X_train.shape[0])
                X_shuffled = X_train[perm]
                Y_shuffled = Y_train[perm]

                step = 0
                for i in range(0, X_train.shape[0], batch_size):
                    step += 1
                    X_batch = X_shuffled[i:i + batch_size]
                    Y_batch = Y_shuffled[i:i + batch_size]

                    sess.run(
                        train_op,
                        feed_dict={x: X_batch, y: Y_batch, is_training: True}
                    )

                    if step % 100 == 0:
                        batch_cost, batch_acc = sess.run(
                            [loss, accuracy],
                            feed_dict={
                                x: X_batch, y: Y_batch, is_training: False
                            }
                        )
                        print("\tStep {}:".format(step))
                        print("\t\tCost: {}".format(batch_cost))
                        print("\t\tAccuracy {}".format(batch_acc))

        save_path = saver.save(sess, save_path)

    return save_path
EOF
