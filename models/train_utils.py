import tensorflow as tf


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, lr, min_ratio, train_steps, warmup_steps=5000):
        super().__init__()

        self.lr = lr
        self.warmup_steps = warmup_steps
        self.train_steps = train_steps
        self.min_lr = lr * min_ratio

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        lr = tf.cond(
            step < self.warmup_steps,
            lambda: self.lr * step / (self.warmup_steps + 1),
            lambda: self.lr * (1 - step / self.train_steps)
        )
        lr = tf.math.maximum(lr, self.min_lr)

        return lr
