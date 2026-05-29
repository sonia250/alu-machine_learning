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

    # Build graph
    x = tf.placeholder(tf.float32, shape=[None, X_train.shape[1]], name='x')
    y = tf.placeholder(tf.float32, shape=[None, Y_train.shape[1]], name='y')

    # Build network with batch normalization
    is_training = tf.placeholder(tf.bool, name='is_training')
    pred = x
    for i, (layer_size, activation) in enumerate(zip(layers, activations)):
        with tf.variable_scope(f'layer_{i+1}'):
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

    # Loss and accuracy
    loss = tf.losses.softmax_cross_entropy(y, pred)
    correct = tf.equal(tf.argmax(y, 1), tf.argmax(pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    # Global step for learning rate decay
    global_step = tf.Variable(0, trainable=False)
    decayed_lr = tf.train.inverse_time_decay(
        alpha, global_step, 1, decay_rate, staircase=True
    )

    # Adam optimizer
    optimizer = tf.train.AdamOptimizer(
        decayed_lr, beta1=beta1, beta2=beta2, epsilon=epsilon
    )
    train_op = optimizer.minimize(loss, global_step=global_step)

    # Saver
    saver = tf.train.Saver()

    # Training
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(epochs + 1):
            # Evaluate on full datasets
            train_cost, train_acc = sess.run(
                [loss, accuracy],
                feed_dict={x: X_train, y: Y_train, is_training: False}
            )
            valid_cost, valid_acc = sess.run(
                [loss, accuracy],
                feed_dict={x: X_valid, y: Y_valid, is_training: False}
            )

            print(f"After {epoch} epochs:")
            print(f"\tTraining Cost: {train_cost}")
            print(f"\tTraining Accuracy: {train_acc}")
            print(f"\tValidation Cost: {valid_cost}")
            print(f"\tValidation Accuracy: {valid_acc}")

            # Train on mini-batches for next epoch
            if epoch < epochs:
                # Shuffle data
                perm = np.random.permutation(X_train.shape[0])
                X_shuffled = X_train[perm]
                Y_shuffled = Y_train[perm]

                step = 0
                for i in range(0, X_train.shape[0], batch_size):
                    step += 1
                    X_batch = X_shuffled[i:i+batch_size]
                    Y_batch = Y_shuffled[i:i+batch_size]

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
                        print(f"\tStep {step}:")
                        print(f"\t\tCost: {batch_cost}")
                        print(f"\t\tAccuracy {batch_acc}")

        # Save model
        save_path = saver.save(sess, save_path)

    return save_path
