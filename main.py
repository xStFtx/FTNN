import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers, callbacks, optimizers

def custom_activation(x):
    return tf.nn.relu(x)  # Replace with your custom function

class Callback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print(f"Starting Epoch {epoch+1}")

    def on_epoch_end(self, epoch, logs=None):
        print(f"Ending Epoch {epoch+1}")

class FourierTransformNeuralNetwork(tf.keras.Model):
    def __init__(self, num_layers, units_per_layer, input_shape, dropout_rate=0.5, regularization_factor=0.01):
        super().__init__()
        self.num_layers = num_layers

        self.layers_list = []
        for _ in range(num_layers):
            self.layers_list.append(layers.Dense(units_per_layer, activation=custom_activation,
                                                 kernel_regularizer=regularizers.l2(regularization_factor)))
            self.layers_list.append(layers.LayerNormalization())  # Using Layer Normalization
            self.layers_list.append(layers.Dropout(dropout_rate))

        self.output_layer = layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        x = tf.reshape(inputs, [-1, np.prod(inputs.shape[1:])])
        for layer in self.layers_list:
            x = layer(x)
        return self.output_layer(x)

# Model instantiation and compilation
model = FourierTransformNeuralNetwork(num_layers=3, units_per_layer=64, input_shape=784)
optimizer = optimizers.Adam(learning_rate=0.001)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Callbacks
lr_scheduler = callbacks.LearningRateScheduler(lambda epoch: 1e-3 * 10 ** (-epoch / 10))
tensorboard_cb = callbacks.TensorBoard(log_dir='./logs')
custom_cb = Callback()  # Custom Callback

callbacks_list = [
    callbacks.EarlyStopping(monitor='val_loss', patience=3),
    callbacks.ModelCheckpoint(filepath='model_{epoch:02d}-{val_loss:.2f}', save_best_only=True, save_format="tf"),
    lr_scheduler,
    tensorboard_cb,
    custom_cb
]

# Data preparation (use actual data here)
train_data = np.random.random((100, 784))
train_labels = np.random.randint(2, size=(100, 1))
test_data = np.random.random((20, 784))
test_labels = np.random.randint(2, size=(20, 1))

# Model training and evaluation
model.fit(train_data, train_labels, epochs=10, validation_split=0.2, callbacks=callbacks_list)
accuracy = model.evaluate(test_data, test_labels)
print(f"Test Accuracy: {accuracy[1]}")
