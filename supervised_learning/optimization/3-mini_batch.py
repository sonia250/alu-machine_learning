#!/usr/bin/env python3
"""Module for training with mini-batch gradient descent"""
import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """
    Trains a loaded neural network model using mini-batch gradient descent

    Args:
        X_train: numpy.ndarray of shape (m, 784) with training data
        Y_train: one-hot numpy.ndarray of shape (m, 10) with training labels
        X_valid: numpy.ndarray of shape (m, 784) with validation data
        Y_valid: one-hot numpy.ndarray of shape (m, 10) with validation labels
        batch_size: number of data points in a batch
        epochs: number of times training passes through the whole dataset
        load_path: path from which to load the model
        save_path: path to where the model should be saved

    Returns:
        the path where the model was saved
    """
    with tf.Session() as sess:
        # Import meta graph and restore session
        saver = tf.train.import_meta_graph(load_path + '.meta')
        saver.restore(sess, load_path)

        # Get tensors and ops from collection
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')[0]

        # Loop over epochs
        for epoch in range(epochs + 1):
            # Calculate training and validation metrics
            train_cost, train_accuracy = sess.run(
                [loss, accuracy],
                feed_dict={x: X_train, y: Y_train}
            )
            valid_cost, valid_accuracy = sess.run(
                [loss, accuracy],
                feed_dict={x: X_valid, y: Y_valid}
            )

            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_accuracy))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_accuracy))

            # Skip mini-batch training on last epoch
            if epoch < epochs:
                # Shuffle training data
                X_shuffled, Y_shuffled = shuffle_data(X_train, Y_train)

                # Loop over batches
                for i in range(0, X_train.shape[0], batch_size):
                    # Get batch
                    X_batch = X_shuffled[i:i + batch_size]
                    Y_batch = Y_shuffled[i:i + batch_size]

                    # Train on batch
                    step_cost, step_accuracy, _ = sess.run(
                        [loss, accuracy, train_op],
                        feed_dict={x: X_batch, y: Y_batch}
                    )

                    # Calculate step number (batch index / batch_size + 1)
                    step = i // batch_size + 1
                    # Print every 100 steps
                    if step % 100 == 0:
                        print("\tStep {}:".format(step))
                        print("\t\tCost: {}".format(step_cost))
                        print("\t\tAccuracy: {}".format(step_accuracy))

        # Save the model
        save_path = saver.save(sess, save_path)

    return save_path
