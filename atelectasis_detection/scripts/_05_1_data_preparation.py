
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

def get_data_generators(
    data_dir='data/processed/', 
    img_size=(224, 224), 
    batch_size=16, 
    validation_split=0.2
):
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        shear_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        validation_split=validation_split
    )

    train_generator = train_datagen.flow_from_directory(
        directory=os.path.join(data_dir, 'after_balancing'),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',       # binary classification
        subset='training',
        shuffle=True
    ) 

    val_datagen = ImageDataGenerator(
        rescale =1./255,
        validation_split = validation_split
    )

    validation_generator = val_datagen.flow_from_directory(
        directory=os.path.join(data_dir, 'after_balancing'),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation',
        shuffle=False
    )

    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        directory=os.path.join(data_dir, 'testing_two_labels_only'),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )

    return train_generator, validation_generator, test_generator


if __name__ == '__main__':
    print("Testing data generators creation...")
    train_gen, val_gen, test_gen = get_data_generators()
    print("\n--- Dataset Summary ---")
    print(f"Training samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    print(f"Testing samples: {test_gen.samples}")
    print(f"Class indices: {train_gen.class_indices}")
