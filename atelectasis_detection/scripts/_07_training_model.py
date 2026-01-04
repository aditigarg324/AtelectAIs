
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau # type: ignore
import os
import matplotlib.pyplot as plt
import pickle

from _05_1_data_preparation import get_data_generators
from _06_model_architecture import create_transfer_learning_model
from datetime import datetime

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

EPOCHS_INITIAL = 15     
EPOCHS_FINE_TUNE = 10
MODEL_SAVE_DIR = 'data/model_outputs'
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

MODEL_FILENAME = f'best_atelectasis_detector_{RUN_ID}.keras'
SAVE_PATH = os.path.join(MODEL_SAVE_DIR, MODEL_FILENAME)

HISTORY_PATH = os.path.join(MODEL_SAVE_DIR, f'training_history_{RUN_ID}.pkl')   
PLOT_PATH = os.path.join(MODEL_SAVE_DIR, 'training_curves_{RUN_ID}.png')      


def train_atelectasis_model():
    """
    Loads data generators, creates the model, defines callbacks, 
    and trains it in two phases: initial training + fine-tuning.
    """

    #Load Data Generators
    print("Loading data generators...")
    train_generator, validation_generator, test_generator = get_data_generators()

    print("\nCreating model architecture (VGG16 Transfer Learning)...")
    model = create_transfer_learning_model()

    
    checkpoint = ModelCheckpoint(
        filepath=SAVE_PATH,
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max'
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5, 
        verbose=1,
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    callbacks_list = [checkpoint, early_stopping, reduce_lr]

    # Phase 1: Initial Training (Frozen VGG16)
    print(f"\n🔹 Stage 1: Training top layers for {EPOCHS_INITIAL} epochs...")
    history_initial = model.fit(
        train_generator,
        epochs=EPOCHS_INITIAL,
        validation_data=validation_generator,
        callbacks=callbacks_list,
        verbose=1
    )

    print("\n✅ Stage 1 complete. Top layers trained.")

    # Phase 2: Fine-Tuning
    print("\n🔹 Stage 2: Fine-tuning last few layers of VGG16...")

    # Unfreeze last 4 convolutional layers in VGG16
    base_model = model.layers[0]  # The VGG16 base model
    for layer in base_model.layers[-8:]: 
        layer.trainable = True

    # Re-compile with a lower learning rate for fine-tuning
    fine_tune_lr = 1e-5
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tune_lr),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    print(f"Fine-tuning {len([l for l in base_model.layers if l.trainable])} VGG16 layers with LR={fine_tune_lr}")

    history_finetune = model.fit(
        train_generator,
        epochs=EPOCHS_FINE_TUNE,
        validation_data=validation_generator,
        callbacks=callbacks_list,
        verbose=1
    )

    print("\n✅ Stage 2 complete. Fine-tuning finished.")
    print(f"📁 Best model weights saved to: {SAVE_PATH}")

    with open(HISTORY_PATH, 'wb') as f:
        pickle.dump((history_initial.history, history_finetune.history), f)
    print(f"📝 Training history saved to: {HISTORY_PATH}")

    # training-curves 
    plot_training_curves(history_initial.history, history_finetune.history)

    return model, (history_initial, history_finetune), test_generator

# Training curves visualization function
def plot_training_curves(history_initial_dict, history_finetune_dict, save_png = True):
    """
    Plots training and validation loss and accuracy for both Stage 1 (top layers)
    and Stage 2 (fine-tuning) on a single combined chart.
    """
    
    acc = history_initial_dict['accuracy'] + history_finetune_dict['accuracy']
    val_acc = history_initial_dict['val_accuracy'] + history_finetune_dict['val_accuracy']
    loss = history_initial_dict['loss'] + history_finetune_dict['loss']
    val_loss = history_initial_dict['val_loss'] + history_finetune_dict['val_loss']

    epochs = range(1, len(acc) + 1)

    # Accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label='Training Accuracy', marker='o')
    plt.plot(epochs, val_acc, label='Validation Accuracy', marker='o')
    plt.axvline(len(history_initial_dict['accuracy']), color='gray', linestyle='--', label='Fine-tuning Start')
    plt.title('Training & Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label='Training Loss', marker='o')
    plt.plot(epochs, val_loss, label='Validation Loss', marker='o')
    plt.axvline(len(history_initial_dict['loss']), color='gray', linestyle='--', label='Fine-tuning Start')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout() 

    if save_png:
        plt.savefig(PLOT_PATH, dpi=300)
        print(f"📊 Training curves saved to: {PLOT_PATH}")

    plt.show()

def load_and_plot_saved_history():
    """
    Utility function to reload and plot the saved history without retraining.
    """
    if not os.path.exists(HISTORY_PATH):
        print("❌ No saved training history found.")
        return
    with open(HISTORY_PATH, 'rb') as f:
        history_initial_dict, history_finetune_dict = pickle.load(f)
    plot_training_curves(history_initial_dict, history_finetune_dict)

if __name__ == '__main__':
    trained_model, (initial_history, finetune_history), final_test_generator = train_atelectasis_model()
    print("\n➡️ Next step: Run 'evaluate_and_report.py' for final evaluation.") 
