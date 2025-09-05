import tensorflow as tf


def limit_gpu_memory() -> None:
    gpu_devices = tf.config.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_memory_growth(device=gpu_devices[0], enable=True)
