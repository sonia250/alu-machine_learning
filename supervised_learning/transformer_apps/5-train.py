#!/usr/bin/env python3
"""Train a Transformer for Portuguese-English translation."""

import tensorflow as tf

Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
Transformer = __import__('5-transformer').Transformer


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Transformer learning-rate schedule."""

    def __init__(self, dm, warmup_steps=4000):
        super().__init__()
        self.dm = tf.cast(dm, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        return tf.math.rsqrt(self.dm) * tf.math.minimum(
            tf.math.rsqrt(step),
            step * self.warmup_steps ** -1.5,
        )


def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    """Train and return a Transformer."""
    data = Dataset(batch_size, max_len)

    input_vocab = data.tokenizer_pt.vocab_size + 2
    target_vocab = data.tokenizer_en.vocab_size + 2

    transformer = Transformer(
        N, dm, h, hidden, input_vocab, target_vocab, max_len
    )

    learning_rate = CustomSchedule(dm)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate,
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-9,
    )

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction='none',
    )

    def loss_function(real, pred):
        loss = loss_object(real, pred)
        mask = tf.cast(tf.not_equal(real, 0), loss.dtype)
        return tf.reduce_sum(loss * mask) / tf.reduce_sum(mask)

    def accuracy_function(real, pred):
        predictions = tf.argmax(pred, axis=2, output_type=real.dtype)
        matches = tf.cast(tf.equal(real, predictions), tf.float32)
        mask = tf.cast(tf.not_equal(real, 0), tf.float32)
        return tf.reduce_sum(matches * mask) / tf.reduce_sum(mask)

    @tf.function
    def train_step(inputs, target):
        decoder_input = target[:, :-1]
        real = target[:, 1:]

        encoder_mask, combined_mask, decoder_mask = create_masks(
            inputs, decoder_input
        )

        with tf.GradientTape() as tape:
            predictions = transformer(
                inputs,
                decoder_input,
                True,
                encoder_mask,
                combined_mask,
                decoder_mask,
            )
            loss = loss_function(real, predictions)

        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(
            zip(gradients, transformer.trainable_variables)
        )
        accuracy = accuracy_function(real, predictions)
        return loss, accuracy

    for epoch in range(epochs):
        epoch_loss = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.Mean()

        for batch, (inputs, target) in enumerate(data.data_train):
            loss, accuracy = train_step(inputs, target)
            epoch_loss.update_state(loss)
            epoch_accuracy.update_state(accuracy)

            if batch % 50 == 0:
                print(
                    f'Epoch {epoch + 1}, batch {batch}: '
                    f'loss {loss.numpy()} accuracy {accuracy.numpy()}'
                )

        print(
            f'Epoch {epoch + 1}: loss {epoch_loss.result().numpy()} '
            f'accuracy {epoch_accuracy.result().numpy()}'
        )

    return transformer
