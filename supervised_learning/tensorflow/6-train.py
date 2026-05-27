#!/usr/bin/env python3
"""Module for training a neural network"""
import tensorflow as tf
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations,
          alpha, iterations, save_path="/tmp/model.ckpt"):
    """
    Builds, trains, and saves a neural network classifier

    Args:
        X_train: numpy.ndarray with training input data
        Y_train: numpy.ndarray with training labels (one-hot)
        X_valid: numpy.ndarray with validation input data
        Y_valid: numpy.ndarray with validation labels (one-hot)
        layer_sizes: list of nodes in each layer
        activations: list of activation functions for each layer
        alpha: learning rate
        iterations: number of iterations to train over
        save_path: path where to save the model

    Returns:
        path where the model was saved
    """
    # Get number of features and classes
    nx = X_train.shape[1]
    classes = Y_train.shape[1]

    # Create graph
    x, y = create_placeholders(nx, classes)

    # Create forward propagation
    y_pred = forward_prop(x, layer_sizes, activations)

    # Calculate loss and accuracy
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)

    # Create training operation
    train_op = create_train_op(loss, alpha)

    # Add to graph collection
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('train_op', train_op)

    # Create saver
    saver = tf.train.Saver()

    # Train the model
    with tf.Session() as sess:
        # Initialize variables
        sess.run(tf.global_variables_initializer())

        # Training loop
        for i in range(iterations + 1):
            # Training data
            train_cost, train_accuracy, _ = sess.run(
                [loss, accuracy, train_op],
                feed_dict={x: X_train, y: Y_train}
            )

            # Validation data
            valid_cost, valid_accuracy = sess.run(
                [loss, accuracy],
                feed_dict={x: X_valid, y: Y_valid}
            )

            # Print every 100 iterations and at 0 and iterations
            if i % 100 == 0 or i == iterations:
                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(train_cost))
                print("\tTraining Accuracy: {}".format(train_accuracy))
                print("\tValidation Cost: {}".format(valid_cost))
                print("\tValidation Accuracy: {}".format(valid_accuracy))

        # Save the model
        save_path = saver.save(sess, save_path)

    return save_path
