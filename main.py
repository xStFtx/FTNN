import tensorflow as tf
from tensorflow.keras import layers, regularizers, models, optimizers
import numpy as np

# Custom Fourier Transform Layer
class FourierTransformLayer(layers.Layer):
    def call(self, inputs):
        return tf.signal.fft(tf.cast(inputs, tf.complex64))

# Custom Complex Dense Layer
class ComplexDense(layers.Layer):
    def __init__(self, units, activation=None, kernel_regularizer=None, **kwargs):
        super(ComplexDense, self).__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)

    def build(self, input_shape):
        real_initializer = tf.keras.initializers.GlorotUniform()
        imag_initializer = tf.keras.initializers.GlorotUniform()
        self.real_kernel = self.add_weight(name='real_kernel', 
                                           shape=(input_shape[-1], self.units),
                                           initializer=real_initializer,
                                           regularizer=self.kernel_regularizer)
        self.imag_kernel = self.add_weight(name='imag_kernel', 
                                           shape=(input_shape[-1], self.units),
                                           initializer=imag_initializer,
                                           regularizer=self.kernel_regularizer)

    def call(self, inputs):
        real = tf.math.real(inputs)
        imag = tf.math.imag(inputs)
        real_part = tf.matmul(real, self.real_kernel) - tf.matmul(imag, self.imag_kernel)
        imag_part = tf.matmul(real, self.imag_kernel) + tf.matmul(imag, self.real_kernel)
        output = tf.complex(real_part, imag_part)
        if self.activation is not None:
            output = self.activation(output)
        return output

# Complex Activation Function
def complex_relu(z):
    real_part = tf.math.maximum(tf.math.real(z), 0)
    imag_part = tf.math.maximum(tf.math.imag(z), 0)
    return tf.complex(real_part, imag_part)

# Custom Loss Function
def complex_loss(targets, predictions):
    targets = tf.cast(targets, dtype=tf.complex64)
    predictions = tf.cast(predictions, dtype=tf.complex64)
    return tf.reduce_mean(tf.square(tf.abs(targets - predictions)))

# Improved Fourier Transform Neural Network with Advanced Features
class ImprovedFTNN(models.Model):
    def __init__(self, num_layers, units_per_layer, input_shape, dropout_rate=0.5, regularization_factor=0.01):
        super().__init__()
        self.num_layers = num_layers

        self.initial_fourier_layer = FourierTransformLayer()
        self.initial_dense_layer = ComplexDense(units_per_layer, activation=complex_relu, kernel_regularizer=regularizers.l2(regularization_factor))

        self.residual_blocks = []
        for _ in range(num_layers):
            layers_list = [
                ComplexDense(units_per_layer, activation=complex_relu, kernel_regularizer=regularizers.l2(regularization_factor)),
                layers.LayerNormalization(),
                layers.LeakyReLU(alpha=0.01),
                layers.Dropout(dropout_rate),
                ComplexDense(units_per_layer, activation=complex_relu)
            ]
            self.residual_blocks.append(layers_list)

        self.output_layer = layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        x = self.initial_fourier_layer(inputs)
        x = self.initial_dense_layer(x)

        for block in self.residual_blocks:
            x_residual = x
            for layer in block:
                x = layer(x)
            x += x_residual

        return self.output_layer(tf.math.abs(x))

# Custom Training Step Function
def train_step(model, inputs, targets, loss_function, optimizer):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = loss_function(targets, predictions)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

# Complete Training and Evaluation Loop
def train_and_evaluate(model, train_data, test_data, test_labels, epochs, optimizer):
    for epoch in range(epochs):
        for inputs, targets in train_data:
            loss = train_step(model, inputs, targets, complex_loss, optimizer)
            print(f"Epoch {epoch + 1}, Loss: {loss.numpy()}")

# Instantiate the model
model = ImprovedFTNN(num_layers=3, units_per_layer=64, input_shape=(784,))
optimizer = optimizers.AdamW(learning_rate=0.001, weight_decay=0.01)  # Using AdamW optimizer

# Sample Data
train_features = np.random.random((100, 784))
train_labels = np.random.random((100, 1))
test_features = np.random.random((20, 784))
test_labels = np.random.random((20, 1))

# Convert to TensorFlow dataset
train_data = tf.data.Dataset.from_tensor_slices((train_features, train_labels))
train_data = train_data.batch(32)

# Training the model
train_and_evaluate(model, train_data, test_features, test_labels, epochs=10, optimizer=optimizer)
