 
from _05_1_data_preparation import get_data_generators

train_gen, val_gen, test_gen = get_data_generators()
x_batch, y_batch = next(train_gen)

print("x_batch shape: ", x_batch.shape)
print("y_batch shape: ", y_batch.shape)

print("min/max pixel values ", x_batch.max(), x_batch.min())
