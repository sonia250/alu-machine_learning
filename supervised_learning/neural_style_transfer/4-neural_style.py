#!/usr/bin/env python3
"""Defines class NST that performs tasks for neural style transfer"""
import numpy as np
import tensorflow as tf


class NST:
    """Performs tasks for Neural Style Transfer"""

    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """Class constructor for Neural Style Transfer class"""
        if type(style_image) is not np.ndarray or \
                len(style_image.shape) != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")

        if type(content_image) is not np.ndarray or \
                len(content_image.shape) != 3:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)")

        style_h, style_w, style_c = style_image.shape
        content_h, content_w, content_c = content_image.shape

        if style_h <= 0 or style_w <= 0 or style_c != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")

        if content_h <= 0 or content_w <= 0 or content_c != 3:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)")

        if (type(alpha) is not float and type(alpha) is not int) or \
                alpha < 0:
            raise TypeError("alpha must be a non-negative number")

        if (type(beta) is not float and type(beta) is not int) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        tf.enable_eager_execution()

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.gram_style_features = []
        self.content_feature = None
        self.load_model()
        self.generate_features()

    @staticmethod
    def scale_image(image):
        """Rescales image to 512 max dimension, pixel values [0, 1]"""
        if type(image) is not np.ndarray or len(image.shape) != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")

        h, w, c = image.shape

        if h <= 0 or w <= 0 or c != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")

        if h > w:
            h_new = 512
            w_new = int(w * (512 / h))
        else:
            w_new = 512
            h_new = int(h * (512 / w))

        resized = tf.image.resize_bicubic(
            np.expand_dims(image, axis=0), size=(h_new, w_new))
        rescaled = resized / 255
        rescaled = tf.clip_by_value(rescaled, 0, 1)

        return rescaled

    def load_model(self):
        """Creates the model used to calculate cost from VGG19"""
        VGG19_model = tf.keras.applications.VGG19(
            include_top=False, weights='imagenet')
        VGG19_model.save("VGG19_base_model")

        custom_objects = {'MaxPooling2D': tf.keras.layers.AveragePooling2D}
        vgg = tf.keras.models.load_model(
            "VGG19_base_model", custom_objects=custom_objects)

        style_outputs = []
        content_output = None

        for layer in vgg.layers:
            if layer.name in self.style_layers:
                style_outputs.append(layer.output)
            if layer.name == self.content_layer:
                content_output = layer.output
            layer.trainable = False

        outputs = style_outputs + [content_output]
        model = tf.keras.models.Model(vgg.input, outputs)

        self.model = model

    @staticmethod
    def gram_matrix(input_layer):
        """Calculates gram matrix of a layer"""
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)):
            raise TypeError(
                "input_layer must be a tensor of rank 4")

        if len(input_layer.shape) != 4:
            raise TypeError(
                "input_layer must be a tensor of rank 4")

        batch, h, w, c = input_layer.shape
        reshaped = tf.reshape(input_layer, [-1, c])
        gram = tf.matmul(tf.transpose(reshaped), reshaped)
        gram = gram / tf.cast(h * w, tf.float32)
        gram = tf.expand_dims(gram, 0)

        return gram

    def generate_features(self):
        """Extracts features used to calculate neural style cost"""
        style_outputs = self.model(self.style_image)
        self.gram_style_features = [
            self.gram_matrix(style_outputs[i])
            for i in range(len(self.style_layers))
        ]

        content_outputs = self.model(self.content_image)
        self.content_feature = content_outputs[-1]

    def layer_style_cost(self, style_output, gram_target):
        """
        Calculates the style cost for a single layer

        Args:
            style_output: tf.Tensor of shape (1, h, w, c) containing the
                layer style output of the generated image
            gram_target: tf.Tensor of shape (1, c, c) the gram matrix of
                the target style output for that layer

        Returns:
            the layer's style cost

        Raises:
            TypeError: if style_output is not a tensor of rank 4
            TypeError: if gram_target shape is incorrect
        """
        if not isinstance(style_output, (tf.Tensor, tf.Variable)):
            raise TypeError(
                "style_output must be a tensor of rank 4")

        if len(style_output.shape) != 4:
            raise TypeError(
                "style_output must be a tensor of rank 4")

        batch, h, w, c = style_output.shape

        if not isinstance(gram_target, (tf.Tensor, tf.Variable)):
            raise TypeError(
                "gram_target must be a tensor of shape [1, {}, {}]".format(
                    c, c))

        if gram_target.shape != (1, c, c):
            raise TypeError(
                "gram_target must be a tensor of shape [1, {}, {}]".format(
                    c, c))

        gram_style = self.gram_matrix(style_output)
        style_cost = tf.reduce_sum(tf.square(gram_style - gram_target))

        return style_cost
