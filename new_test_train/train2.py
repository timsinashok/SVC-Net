# This script is used to train the SVC-Net model with 3N input.
# It is based on the train.py script, but with 3N input.
# it uses SSIM loss function.

import tensorflow as tf
import numpy as np
import os
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping

from tensorflow.keras.callbacks import TensorBoard
import datetime


from keras.saving import register_keras_serializable


# Directory for TensorBoard logs
log_dir = "/scratch/netid/Lab-PI/october/results/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

from model import *
from custom_utils import *

# Configuration
batch_size = 16 # Adjust batch size based on GPU memory availability
epochs = 100
# image_height, image_width = 512, 512  # Image dimensions
image_height, image_width = 512, 192  # Image dimensions
validation_split = 0.2  # 20% data for validation

# Define paths
oct_root_path = Path("/scratch/netid/Lab-PI/Professor_Data/OCT")  # Root directory for OCT images
octa_root_path = Path("/scratch/netid/Lab-PI/Professor_Data/OCTA")  # Root directory for OCTA images
export_dir = Path("/scratch/netid/Lab-PI/october/results/3n_new4")
if not os.path.exists(export_dir):
    os.makedirs(export_dir)

# Model checkpoint configuration
model_file_format = os.path.join(export_dir, "svcnet_model.{epoch:03d}.weights.h5")
checkpointer = ModelCheckpoint(model_file_format, save_best_only=True, save_weights_only=True)

# 3N data generator
class SVCNetDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, oct_root_path, octa_root_path, batch_size, image_size, paired_subdirs):
        self.oct_root_path = Path(oct_root_path)
        self.octa_root_path = Path(octa_root_path)
        self.batch_size = batch_size
        self.image_size = image_size

        self.oct_images = []
        self.octa_images = []

        for oct_subdir, octa_subdir in paired_subdirs:
            oct_dir = self.oct_root_path / oct_subdir
            octa_dir = self.octa_root_path / octa_subdir

            oct_files = sorted(list(oct_dir.glob("*.jpg")))
            octa_files = sorted(list(octa_dir.glob("*.jpg")))

            print(f"OCT files in {oct_dir} is {len(oct_files)}")
            print(f"OCTA files in {octa_dir} is {len(octa_files)}")

            min_len = min(len(oct_files), len(octa_files))

            if min_len < 3:
                print(f"Skipping pair ({oct_subdir}, {octa_subdir}) — not enough images.")
                continue

            # Trim both lists to the same length
            self.oct_images.extend(oct_files[:min_len])
            self.octa_images.extend(octa_files[:min_len])

        # Use the smallest length for indexing
        min_len = min(len(self.oct_images), len(self.octa_images))
        self.indices = list(range(min_len - 2))

        if len(self.indices) < self.batch_size:
            print("Warning: Not enough data to form a full batch.")

    def __len__(self):
        return len(self.indices) // self.batch_size

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_oct = []
        batch_octa = []

        for i in batch_indices:
            oct_triplet = [self.load_image(str(self.oct_images[i + j]), gray=True) for j in range(3)]
            concatenated_oct = np.concatenate(oct_triplet, axis=-1)
            octa_image = self.load_image(str(self.octa_images[i + 1]), gray=True)

            batch_oct.append(concatenated_oct)
            batch_octa.append(octa_image)

        return np.array(batch_oct), np.array(batch_octa)

    def load_image(self, path, gray=False):
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
        image = cv2.resize(image, self.image_size)
        if gray:
            image = np.expand_dims(image, axis=-1)
        return image.astype(np.float32) / 255.0

# 5N Data generator
class SVCNetDataGenerator5N(tf.keras.utils.Sequence):
    def __init__(self, oct_root_path, octa_root_path, batch_size, image_size, paired_subdirs):
        self.oct_root_path = Path(oct_root_path)
        self.octa_root_path = Path(octa_root_path)
        self.batch_size = batch_size
        self.image_size = image_size

        self.oct_images = []
        self.octa_images = []

        for oct_subdir, octa_subdir in paired_subdirs:
            oct_dir = self.oct_root_path / oct_subdir
            octa_dir = self.octa_root_path / octa_subdir

            oct_files = sorted(list(oct_dir.glob("*.jpg")))
            octa_files = sorted(list(octa_dir.glob("*.jpg")))

            print(f"OCT files in {oct_dir} is {len(oct_files)}")
            print(f"OCTA files in {octa_dir} is {len(octa_files)}")

            min_len = min(len(oct_files), len(octa_files))

            if min_len < 5:  # need at least 5 OCT + 1 OCTA
                print(f"Skipping pair ({oct_subdir}, {octa_subdir}) — not enough images.")
                continue

            # Trim both lists to the same length
            self.oct_images.extend(oct_files[:min_len])
            self.octa_images.extend(octa_files[:min_len])

        # Use the smallest length for indexing
        min_len = min(len(self.oct_images), len(self.octa_images))
        # we stop at min_len-4 because we need i...i+4
        self.indices = list(range(min_len - 4))

        if len(self.indices) < self.batch_size:
            print("Warning: Not enough data to form a full batch.")

    def __len__(self):
        return len(self.indices) // self.batch_size

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_oct = []
        batch_octa = []

        for i in batch_indices:
            # 5 consecutive OCT slices
            oct_pent = [self.load_image(str(self.oct_images[i + j]), gray=True) for j in range(5)]
            concatenated_oct = np.concatenate(oct_pent, axis=-1)

            # OCTA target is the central slice (aligned with i+2)
            octa_image = self.load_image(str(self.octa_images[i + 2]), gray=True)

            batch_oct.append(concatenated_oct)
            batch_octa.append(octa_image)

        return np.array(batch_oct), np.array(batch_octa)

    def load_image(self, path, gray=False):
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
        image = cv2.resize(image, self.image_size)
        if gray:
            image = np.expand_dims(image, axis=-1)
        return image.astype(np.float32) / 255.0



# Learning rate scheduler function
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

# Early stopping configuration
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)


print("Loading data directories")
# Get list of subdirectories in OCT and OCTA root directories
oct_subdirs = sorted([d.name for d in oct_root_path.iterdir() if d.is_dir()])
octa_subdirs = sorted([d.name for d in octa_root_path.iterdir() if d.is_dir()])

print("OCT Subdirs:")
print(oct_subdirs)
print("OCTA Subdirs:")
print(octa_subdirs)

# Clean subdir names and remove spaces
oct_subdirs = sorted([d.name.strip().replace(" ", "") for d in oct_root_path.iterdir() if d.is_dir()])
octa_subdirs = sorted([d.name.strip().replace(" ", "") for d in octa_root_path.iterdir() if d.is_dir()])

# Extract base IDs
oct_ids = sorted(set(s[:3] for s in oct_subdirs if "OCT" in s))
octa_ids = sorted(set(s[:3] for s in octa_subdirs if "Speckle" in s))
common_ids = sorted(set(oct_ids).intersection(octa_ids))

print("Common subject IDs:", common_ids)

# Split train/val
train_ids, val_ids = train_test_split(common_ids, test_size=validation_split, random_state=42)

# Build subdir pairs
def generate_pairs(ids):
    return [
        *[(f"{id}_XOCT_Images", f"{id}_XSpeckle_Denoised") for id in ids],
        *[(f"{id}_YOCT_Images", f"{id}_YSpeckle_Denoised") for id in ids]
    ]

train_pairs = generate_pairs(train_ids)
val_pairs = generate_pairs(val_ids)

print(f"Train Pairs = {train_pairs}")
print(f"Val Pairs = {val_pairs}")

# Build data generators
train_generator = SVCNetDataGenerator(
    oct_root_path, octa_root_path,
    batch_size=batch_size,
    image_size=(image_width, image_height),
    paired_subdirs=train_pairs
)

val_generator = SVCNetDataGenerator(
    oct_root_path, octa_root_path,
    batch_size=batch_size,
    image_size=(image_width, image_height),
    paired_subdirs=val_pairs
)




# Load the model
model = svcnet_model(height=image_height, width=image_width, n_channels=3)


@register_keras_serializable()
def SSIMLoss(y_true, y_pred):
    """
    Structural similarity index measure (SSIM) loss function

    Args:
        y_true: ground truth tensor
        y_pred: prediction tensor by the model

    Returns:
        loss
    """
    return (1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0)))

print("Compiling")
# Compile the model
model.compile(
        optimizer=Adam(learning_rate=0.0003),
        loss=SSIMLoss,
        metrics=[tf.keras.metrics.MeanSquaredError(name='mse'), tf.keras.metrics.MeanAbsoluteError(name='mae')]
    )

# Train model
model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator,
    callbacks=[checkpointer, LearningRateScheduler(scheduler), early_stopping]
)

# Save the final model
final_model_path = os.path.join(export_dir, "revised_size_final_svcnet_model.hdf5")
model.save(final_model_path)
print(f"Final model saved to: {final_model_path}")

# Save the training history
history_file_path = os.path.join(export_dir, "revised_size_training_history_lr_scheduled.npy")
np.save(history_file_path,model.history)
print(f"Training history saved to: {history_file_path}")
