import tensorflow as tf

physical_devices = tf.config.list_physical_devices()
print("Physical Devices:", physical_devices)

gpus = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpus))
print("GPUs: ", gpus)
