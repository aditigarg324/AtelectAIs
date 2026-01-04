import tensorflow as tf
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.applications import VGG16 # type: ignore
from tensorflow.keras import regularizers # type: ignore


IMAGE_SIZE = (224, 224)
INPUT_SHAPE = IMAGE_SIZE + (3,)
LEARNING_RATE = 1e-4
METRICS = ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]

def create_transfer_learning_model():
    """
    Creates a atelectasis detection model using Transfer Learning (VGG16),
    with regularization and dropout to reduce overfitting. 
    """
    base_model = VGG16(
        input_shape=INPUT_SHAPE,
        include_top=False,
        weights='imagenet'
    )

    # Freeze all convolutional layers initially
    for layer in base_model.layers:
        layer.trainable = False

    # Classification head (smaller to reduce overfitting)
    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation='relu', kernel_regularizer = regularizers.l2(1e-4)),
        BatchNormalization(),
        Dropout(0.6),
        Dense(64, activation='relu', kernel_regularizer = regularizers.l2(1e-4)),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=METRICS
    )

    model.summary()
    return model

if __name__ == '__main__':
    pneumonia_model = create_transfer_learning_model()
    print("\nModel architecture defined successfully.")



