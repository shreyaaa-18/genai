
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# 1. Generate real data (points forming a ring)
rng = np.random.default_rng(42)

def real_batch(n):
    angles = rng.uniform(0, 2 * np.pi, n)
    r = 0.8 + 0.05 * rng.normal(size=n)
    x = np.stack([r * np.cos(angles),
                  r * np.sin(angles)], axis=1)
    return x.astype("float32")


# 2. Define dimensions
LATENT_DIM = 2
DATA_DIM = 2
HIDDEN = 32
BATCH = 128


# 3. Build Generator
def build_generator():
    return keras.Sequential([
        layers.Input(shape=(LATENT_DIM,)),
        layers.Dense(HIDDEN, activation="leaky_relu"),
        layers.Dense(HIDDEN, activation="leaky_relu"),
        layers.Dense(DATA_DIM, activation="tanh")
    ])


# 4. Build Discriminator
def build_discriminator():
    return keras.Sequential([
        layers.Input(shape=(DATA_DIM,)),
        layers.Dense(HIDDEN, activation="leaky_relu"),
        layers.Dense(HIDDEN, activation="leaky_relu"),
        layers.Dense(1, activation="sigmoid")
    ])


G = build_generator()
D = build_discriminator()


# 5. Define loss and optimizers
bce = keras.losses.BinaryCrossentropy()

g_optimizer = keras.optimizers.Adam(
    learning_rate=0.002, beta_1=0.5
)

d_optimizer = keras.optimizers.Adam(
    learning_rate=0.002, beta_1=0.5
)


# 6. Train the GAN
@tf.function
def train_step(real):

    # Train Discriminator
    z = tf.random.normal((BATCH, LATENT_DIM))

    with tf.GradientTape() as tape:
        fake = G(z, training=True)

        real_pred = D(real, training=True)
        fake_pred = D(fake, training=True)

        d_loss = (
            bce(tf.ones_like(real_pred), real_pred)
            + bce(tf.zeros_like(fake_pred), fake_pred)
        )

    d_grads = tape.gradient(d_loss, D.trainable_variables)
    d_optimizer.apply_gradients(
        zip(d_grads, D.trainable_variables)
    )

    # Train Generator
    z = tf.random.normal((BATCH, LATENT_DIM))

    with tf.GradientTape() as tape:
        fake = G(z, training=True)
        fake_pred = D(fake, training=True)

        g_loss = bce(
            tf.ones_like(fake_pred), fake_pred
        )

    g_grads = tape.gradient(g_loss, G.trainable_variables)
    g_optimizer.apply_gradients(
        zip(g_grads, G.trainable_variables)
    )

    return d_loss, g_loss


# 7. Run training
EPOCHS = 4000

for epoch in range(EPOCHS):
    real = tf.convert_to_tensor(real_batch(BATCH))
    d_loss, g_loss = train_step(real)

    if epoch % 500 == 0:
        print(
            f"Epoch {epoch}: "
            f"D Loss = {float(d_loss):.4f}, "
            f"G Loss = {float(g_loss):.4f}"
        )


# 8. Generate fake samples using trained Generator
z = tf.random.normal((500, LATENT_DIM))
generated = G(z, training=False).numpy()

# 9. Plot generated samples
plt.scatter(generated[:, 0], generated[:, 1], s=8)
plt.title("GAN Generated Samples")
plt.xlabel("X1")
plt.ylabel("X2")
plt.axis("equal")
plt.show()
