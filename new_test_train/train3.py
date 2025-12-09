# This script is used to train the SVC-Net model with 5N input.
# It is based on the train2.py script, but with 5N input.
# it uses composite loss function which is a combination of SSIM loss and MAE loss.

import tensorflow as tf
import numpy as np
import os
import cv2
import datetime
from pathlib import Path
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
from sklearn.model_selection import train_test_split

from model import svcnet_model


# ==== CONFIG ====
batch_size = 16
epochs = 200
image_height, image_width = 512, 192
validation_split = 0.2

oct_root_path = Path("/scratch/netid/Lab-PI/Professor_Data/OCT")
octa_root_path = Path("/scratch/netid/Lab-PI/Professor_Data/OCTA")
export_dir = Path("/scratch/netid/Lab-PI/october/results/improved_v2_5n")
export_dir.mkdir(parents=True, exist_ok=True)


# ==== LOGGING ====
log_dir = export_dir / f"logs/fit/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
tensorboard_callback = TensorBoard(log_dir=str(log_dir), histogram_freq=1)


# ==== LOSS ====
@tf.keras.utils.register_keras_serializable()
def composite_loss(y_true, y_pred):
    ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
    mae_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
    return 0.85 * ssim_loss + 0.15 * mae_loss


# Data generators
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


# ==== MODEL ====
model = svcnet_model(height=image_height, width=image_width, n_channels=5)

# Cosine Decay LR
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


# ==== DATA ====
from sklearn.model_selection import train_test_split

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

train_generator = SVCNetDataGenerator5N(oct_root_path, octa_root_path, batch_size=batch_size,
                                      image_size=(image_width, image_height), paired_subdirs=train_pairs)
val_generator = SVCNetDataGenerator5N(oct_root_path, octa_root_path, batch_size=batch_size,
                                    image_size=(image_width, image_height), paired_subdirs=val_pairs)


# ==== TRAIN ====
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    callbacks=[checkpointer, early_stopping, tensorboard_callback],
    verbose=1
)

final_model_path = export_dir / "final_svcnet_model.hdf5"
model.save(final_model_path)
print(f"✅ Final model saved to: {final_model_path}")

np.save(str(export_dir / "training_history.npy"), history.history)
print("✅ Training history saved.")
