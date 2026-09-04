#!/usr/bin/env python3
"""Module for Neural Style Transfer"""
import numpy as np
import tensorflow as tf


class NST:
    """Performs tasks for neural style transfer"""

    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Initialize Neural Style Transfer

        Args:
            style_image: numpy.ndarray of shape (h, w, 3) for style reference
            content_image: numpy.ndarray of shape (h, w, 3) for content reference
            alpha: weight for content cost (default 1e4)
            beta: weight for style cost (default 1)

        Raises:
            TypeError: if inputs are invalid
        """
        # Validate style_image
        if (not isinstance(style_image, np.ndarray) or
                style_image.ndim != 3 or style_image.shape[2] != 3):
            raise TypeError(
                'style_image must be a numpy.ndarray with shape (h, w, 3)'
            )

        # Validate content_image
        if (not isinstance(content_image, np.ndarray) or
                content_image.ndim != 3 or content_image.shape[2] != 3):
            raise TypeError(
                'content_image must be a numpy.ndarray with shape (h, w, 3)'
            )

        # Validate alpha
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError('alpha must be a non-negative number')

        # Validate beta
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError('beta must be a non-negative number')

        # Set TensorFlow to execute eagerly
        tf.compat.v1.enable_eager_execution()

        # Preprocess images
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = float(alpha)
        self.beta = float(beta)

    @staticmethod
    def scale_image(image):
        """
        Rescales an image such that pixel values are between 0 and 1
        and its largest side is 512 pixels

        Args:
            image: numpy.ndarray of shape (h, w, 3) to be scaled

        Returns:
            scaled image as tf.tensor with shape (1, h_new, w_new, 3)

        Raises:
            TypeError: if image is not valid
        """
        if (not isinstance(image, np.ndarray) or
                image.ndim != 3 or image.shape[2] != 3):
            raise TypeError(
                'image must be a numpy.ndarray with shape (h, w, 3)'
            )

        # Get original dimensions
        h, w = image.shape[0], image.shape[1]

        # Calculate new dimensions maintaining aspect ratio
        # Max dimension becomes 512
        if h > w:
            new_h = 512
            new_w = int(w * (512.0 / h))
        else:
            new_w = 512
            new_h = int(h * (512.0 / w))

        # Convert to float32 tensor and add batch dimension
        img_tensor = tf.cast(image, tf.float32)
        img_tensor = tf.expand_dims(img_tensor, 0)

        # Resize using bicubic interpolation
        img_tensor = tf.image.resize(
            img_tensor,
            [new_h, new_w],
            method='bicubic'
        )

        # Normalize pixel values from [0, 255] to [0, 1]
        img_tensor = img_tensor / 255.0

        return img_tensor