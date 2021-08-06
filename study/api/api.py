import tensorflow_datasets as tfds


train_datasets = tfds.load('cifar10', split='train')
valid_datasets = tfds.load('cifar10', split='test')

print(train_datasets)

for data in train_datasets.take(5):
    print(data['image'])
    print(data['label'])


tf.cast(data['image'], tf.float32) / 255.0

