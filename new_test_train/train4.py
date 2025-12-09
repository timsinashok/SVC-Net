# This script is used to train the SVC-Net model with data augmentation.
# It is based on the train2.py script, but with data augmentation.
# it uses composite loss function which is a combination of SSIM loss and MAE loss.


import tensorflow as tf
import numpy as np
import os
import datetime
from pathlib import Path
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts

from model import svcnet_model
from train2 import SVCNetDataGenerator
from sklearn.model_selection import train_test_split
import tensorflow_addons as tfa



# ==== CONFIG ====
batch_size = 16
epochs = 200
image_height, image_width = 512, 192
validation_split = 0.1

oct_root_path = Path("/scratch/netid/Lab-PI/Professor_Data/OCT")
octa_root_path = Path("/scratch/netid/Lab-PI/Professor_Data/OCTA")
export_dir = Path("/scratch/netid/Lab-PI/october/results/improved_v2_aug")
export_dir.mkdir(parents=True, exist_ok=True)

log_dir = export_dir / f"logs/fit/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
tensorboard_callback = TensorBoard(log_dir=str(log_dir), histogram_freq=1)


# ==== LOSS ====
@tf.keras.utils.register_keras_serializable()
def composite_loss(y_true, y_pred):
    ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
    mae_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
    return 0.85 * ssim_loss + 0.15 * mae_loss


# ==== AUGMENTATION WRAPPER ====
def augment_batch(x_batch, y_batch):
    # Use tf.image for GPU-based augmentation
    x_batch = tf.image.random_flip_left_right(x_batch)
    y_batch = tf.image.random_flip_left_right(y_batch)

    x_batch = tf.image.random_brightness(x_batch, max_delta=0.05)
    y_batch = tf.image.random_brightness(y_batch, max_delta=0.05)

    # Random small rotations
    angle = tf.random.uniform([], -0.05, 0.05)
    x_batch = tfa.image.rotate(x_batch, angles=angle, interpolation='BILINEAR')
    y_batch = tfa.image.rotate(y_batch, angles=angle, interpolation='BILINEAR')

    return x_batch, y_batch


class AugmentedDataGenerator(SVCNetDataGenerator):
    def __getitem__(self, idx):
        x_batch, y_batch = super().__getitem__(idx)
        x_batch, y_batch = augment_batch(x_batch, y_batch)
        return x_batch, y_batch


# ==== DATA ====
oct_subdirs = sorted([d.name.strip().replace(" ", "") for d in oct_root_path.iterdir() if d.is_dir()])
octa_subdirs = sorted([d.name.strip().replace(" ", "") for d in octa_root_path.iterdir() if d.is_dir()])

oct_ids = sorted(set(s[:3] for s in oct_subdirs if "OCT" in s))
octa_ids = sorted(set(s[:3] for s in octa_subdirs if "Speckle" in s))
common_ids = sorted(set(oct_ids).intersection(octa_ids))

train_ids, val_ids = train_test_split(common_ids, test_size=validation_split, random_state=42)

def generate_pairs(ids):
    return [
        *[(f"{id}_XOCT_Images", f"{id}_XSpeckle_Denoised") for id in ids],
        *[(f"{id}_YOCT_Images", f"{id}_YSpeckle_Denoised") for id in ids]
    ]

train_pairs = generate_pairs(train_ids)
val_pairs = generate_pairs(val_ids)

train_generator = AugmentedDataGenerator(oct_root_path, octa_root_path, batch_size=batch_size,
                                         image_size=(image_width, image_height), paired_subdirs=train_pairs)
val_generator = SVCNetDataGenerator(oct_root_path, octa_root_path, batch_size=batch_size,
                                    image_size=(image_width, image_height), paired_subdirs=val_pairs)


# ==== MODEL ====
model = svcnet_model(height=image_height, width=image_width, n_channels=3)

initial_lr = 0.0003
lr_schedule = CosineDecayRestarts(initial_learning_rate=initial_lr, first_decay_steps=30)
optimizer = Adam(learning_rate=lr_schedule)

model.compile(optimizer=optimizer, loss=composite_loss,
              metrics=[tf.keras.metrics.MeanSquaredError(name='mse'),
                       tf.keras.metrics.MeanAbsoluteError(name='mae')])


# ==== CALLBACKS ====
model_file_format = str(export_dir / "svcnet_model.{epoch:03d}.weights.h5")
checkpointer = ModelCheckpoint(model_file_format, save_best_only=True, save_weights_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)


# ==== TRAIN ====
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    callbacks=[checkpointer, early_stopping, tensorboard_callback],
    verbose=1
)

# final_model_path = export_dir / "final_svcnet_model.hdf5"
# model.save(final_model_path)
# np.save(str(export_dir / "training_history.npy"), history.history)

# print(f"✅ Final model saved to: {final_model_path}")
# print("✅ Training history saved.")


# Save the final model
final_model_path = os.path.join(export_dir, "final_svcnet_model.hdf5")
model.save(final_model_path)
print(f"Final model saved to: {final_model_path}")

# Save the training history
history_file_path = os.path.join(export_dir, "training_history.npy")
np.save(history_file_path,model.history)
print(f"Training history saved to: {history_file_path}")