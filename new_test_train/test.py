# This script is used to test the SVC-Net model with 3N input.
# it uses SSIM loss function.

import tensorflow as tf
import numpy as np
import os
import cv2
from pathlib import Path
from model import *
from keras.saving import register_keras_serializable

# Config
batch_size = 16
# image_height, image_width = 512, 512
image_height, image_width = 512, 160

# Paths
oct_root_path = Path("/scratch/netid/Lab-PI/Professor_Data/Test-Data2/OCT")
octa_root_path = Path("/scratch/netid/Lab-PI/Professor_Data/Test-Data2/OCTA")
export_dir = Path("/scratch/netid/Lab-PI/october/results/improved_v2")
final_model_path = export_dir / "final_svcnet_model.hdf5" #"revised_size_final_svcnet_model.hdf5"

@register_keras_serializable()
def SSIMLoss(y_true, y_pred):
    return (1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0)))

model = tf.keras.models.load_model(final_model_path, custom_objects={'SSIMLoss': SSIMLoss})

# 3n data generator
class SVCNetTestDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, oct_root_path, octa_root_path, batch_size, image_size, paired_subdirs):
        self.oct_root_path = Path(oct_root_path)
        self.octa_root_path = Path(octa_root_path)
        self.batch_size = batch_size
        self.image_size = image_size

        self.oct_images = []
        self.octa_images = []

        for oct_subdir, octa_subdir in paired_subdirs:
            print("oct_subdir")
            print("octa_subdir")
            oct_dir = self.oct_root_path / oct_subdir
            octa_dir = self.octa_root_path / octa_subdir

            oct_files = sorted(list(oct_dir.glob("*.jpg")))
            octa_files = sorted(list(octa_dir.glob("*.jpg")))

            min_len = min(len(oct_files), len(octa_files))

            if min_len < 3:
                print(f"Skipping ({oct_subdir}, {octa_subdir}) — not enough images: OCT={len(oct_files)}, OCTA={len(octa_files)}")
                continue

            self.oct_images.extend(oct_files[:min_len])
            self.octa_images.extend(octa_files[:min_len])

        self.indices = list(range(len(self.oct_images) - 2))
        if len(self.indices) < self.batch_size:
            print("⚠️ Warning: Not enough data to form a full batch. Will still proceed.")

    def __len__(self):
        return max(1, len(self.indices) // self.batch_size)

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_oct, batch_octa = [], []

        for i in batch_indices:
            oct_triplet = [self.load_image(self.oct_images[i + j], gray=True) for j in range(3)]
            concatenated_oct = np.concatenate(oct_triplet, axis=-1)
            octa_image = self.load_image(self.octa_images[i + 1], gray=True)

            batch_oct.append(concatenated_oct)
            batch_octa.append(octa_image)

        return np.array(batch_oct), np.array(batch_octa)

    def load_image(self, path, gray=False):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
        image = cv2.resize(image, self.image_size)
        if gray:
            image = np.expand_dims(image, axis=-1)
        return image.astype(np.float32) / 255.0

# 5N Data generator
# class SVCNetDataGenerator(tf.keras.utils.Sequence):
#     def __init__(self, oct_root_path, octa_root_path, batch_size, image_size, paired_subdirs):
#         self.oct_root_path = Path(oct_root_path)
#         self.octa_root_path = Path(octa_root_path)
#         self.batch_size = batch_size
#         self.image_size = image_size

#         self.oct_images = []
#         self.octa_images = []

#         for oct_subdir, octa_subdir in paired_subdirs:
#             oct_dir = self.oct_root_path / oct_subdir
#             octa_dir = self.octa_root_path / octa_subdir

#             oct_files = sorted(list(oct_dir.glob("*.png")))
#             octa_files = sorted(list(octa_dir.glob("*.png")))

#             print(f"OCT files in {oct_dir} is {len(oct_files)}")
#             print(f"OCTA files in {octa_dir} is {len(octa_files)}")

#             min_len = min(len(oct_files), len(octa_files))

#             if min_len < 5:  # need at least 5 OCT + 1 OCTA
#                 print(f"Skipping pair ({oct_subdir}, {octa_subdir}) — not enough images.")
#                 continue

#             # Trim both lists to the same length
#             self.oct_images.extend(oct_files[:min_len])
#             self.octa_images.extend(octa_files[:min_len])

#         # Use the smallest length for indexing
#         min_len = min(len(self.oct_images), len(self.octa_images))
#         # we stop at min_len-4 because we need i...i+4
#         # self.indices = list(range(min_len - 4))
        
        
#         # Edited
#         # After
#         self.indices = list(range(len(self.oct_images) - 4))

#         if len(self.indices) < self.batch_size:
#             print("Warning: Not enough data to form a full batch.")

#     # def __len__(self):
#     #     return len(self.indices) // self.batch_size

#     def __len__(self):
#         return max(1, len(self.indices) // self.batch_size)

#     def __getitem__(self, idx):
#         batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]

#         batch_oct = []
#         batch_octa = []

#         for i in batch_indices:
#             # 5 consecutive OCT slices
#             oct_pent = [self.load_image(str(self.oct_images[i + j]), gray=True) for j in range(5)]
#             concatenated_oct = np.concatenate(oct_pent, axis=-1)

#             # OCTA target is the central slice (aligned with i+2)
#             octa_image = self.load_image(str(self.octa_images[i + 2]), gray=True)

#             batch_oct.append(concatenated_oct)
#             batch_octa.append(octa_image)

#         return np.array(batch_oct), np.array(batch_octa)

#     def load_image(self, path, gray=False):
#         image = cv2.imread(path, cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
#         image = cv2.resize(image, self.image_size)
#         if gray:
#             image = np.expand_dims(image, axis=-1)
#         return image.astype(np.float32) / 255.0





# --- Load test subdirs and generate valid pairs ---
print("🔍 Loading test subdirectories...")

oct_subdirs = sorted([d.name.strip().replace(" ", "") for d in oct_root_path.iterdir() if d.is_dir()])
octa_subdirs = sorted([d.name.strip().replace(" ", "") for d in octa_root_path.iterdir() if d.is_dir()])

print(oct_subdirs)
print(octa_subdirs)

oct_ids = sorted(set(s[:3] for s in oct_subdirs if "OCT" in s))
octa_ids = sorted(set(s[:3] for s in octa_subdirs if "Speckle" in s))
common_ids = sorted(set(oct_ids).intersection(octa_ids))

print(f"🧪 Found common IDs: {common_ids}")

def generate_pairs(ids):
    # return [
    #     (f"{id}_XOCT", f"{id}_XSpeckle") for id in ids
    # ] + [
    #     (f"{id}_YOCT", f"{id}_YSpeckle") for id in ids
    # ] +
    return[
        # *[(f"{id}_XOCT_Images", f"{id}_XSpeckle_Denoised") for id in ids],
        *[(f"{id}_YOCT_Images", f"{id}_YSpeckle_Denoised") for id in ids]
    ]

test_pairs = generate_pairs(common_ids)
print(f"🔗 Candidate test pairs: {len(test_pairs)}")

# Create the generator
test_generator = SVCNetTestDataGenerator(
    oct_root_path, octa_root_path,
    batch_size=batch_size,
    image_size=(image_width, image_height),
    paired_subdirs=test_pairs
)

if len(test_generator) == 0:
    raise RuntimeError("🚫 No valid test data available. Check image folders and naming consistency.")

print(f"✅ Test generator ready with {len(test_generator)} batches")

# --- Evaluate model ---
test_loss, test_mse, test_mae = model.evaluate(test_generator)
print(f"📊 Test Results → Loss: {test_loss:.4f}, MSE: {test_mse:.4f}, MAE: {test_mae:.4f}")

# --- Predict and save ---
print("📸 Generating predictions...")
predictions = model.predict(test_generator)

predictions_dir = export_dir / "predictions2/improved_v2"
predictions_dir.mkdir(parents=True, exist_ok=True)

for i, pred in enumerate(predictions):
    pred_image = (pred * 255).astype(np.uint8).squeeze()
    pred_path = predictions_dir / f"pred_{i}.png"
    cv2.imwrite(str(pred_path), pred_image)

print(f"✅ Saved {len(predictions)} predictions to: {predictions_dir}")





# # import tensorflow as tf
# # import numpy as np
# # import os
# # import cv2
# # from pathlib import Path
# # from model import *
# # # from custom_utils import *

# # # Configuration
# # batch_size = 4  # Adjust batch size based on GPU memory availability
# # image_height, image_width = 800, 320  # Image dimensions

# # # Define paths
# # #oct_root_path = Path("/scratch/at5282/Lab-PI/OCTA500/Test-Data/OCT")  # Root directory for OCT images
# # #octa_root_path = Path("/scratch/at5282/Lab-PI/OCTA500/Test-Data/OCTA")  # Root directory for OCTA images
# # export_dir = Path("/scratch/at5282/Lab-PI/OCTA500/Results")

# # oct_root_path = Path("/scratch/at5282/Lab-PI/Fingernail/Data/Test-Data/OCT")  # Root directory for OCT images
# # octa_root_path = Path("/scratch/at5282/Lab-PI/Fingernail/Data/Test-Data/OCTA")  # Root directory for OCTA images
# # export_dir = Path("/scratch/at5282/Lab-PI/New_Results")

# # # Model checkpoint configuration
# # # final_model_path = os.path.join(export_dir, "/revised_size_final_svcnet_model.hdf5")
# # final_model_path = "/scratch/at5282/Lab-PI/New_Results/revised_size_final_svcnet_model.hdf5"

# # @tf.keras.utils.register_keras_serializable()
# # def SSIMLoss(y_true, y_pred):
# #     """
# #     Structural similarity index measure (SSIM) loss function

# #     Args:
# #         y_true: ground truth tensor
# #         y_pred: prediction tensor by the model

# #     Returns:
# #         loss
# #     """
# #     return (1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0)))

# # # Load the model
# # model = tf.keras.models.load_model(final_model_path, custom_objects={'SSIMLoss': SSIMLoss})
# # # model.summary()

# # class SVCNetTestDataGenerator(tf.keras.utils.Sequence):
# #     def __init__(self, oct_root_path, octa_root_path, batch_size, image_size, subdir_list):
# #         self.oct_root_path = Path(oct_root_path)
# #         self.octa_root_path = Path(octa_root_path)
# #         self.batch_size = batch_size
# #         self.image_size = image_size

# #         self.subdir_list = subdir_list
# #         self.oct_images = []
# #         self.octa_images = []

# #         for subdir in self.subdir_list:
# #             oct_subdir = self.oct_root_path / subdir
# #             octa_subdir = self.octa_root_path / subdir

# #             oct_files = sorted([str(p) for p in oct_subdir.glob("*.png")])
# #             octa_files = sorted([str(p) for p in octa_subdir.glob("*.png")])

# #             if (len(oct_files) == len(octa_files)) :  # editing to accommodate for testing example 
# #                 self.oct_images.extend(oct_files)
# #                 self.octa_images.extend(octa_files)
# #             else:
# #                 print(f"Skipping {subdir} as it does not have the required number of images.")

# #         self.indices = list(range(len(self.oct_images) - 2))

# #         if len(self.indices) < self.batch_size:
# #             print("Warning: Not enough data to form a batch.")

# #     def __len__(self):
# #         return len(self.indices) // self.batch_size

# #     def __getitem__(self, idx):
# #         batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]

# #         batch_oct = []
# #         batch_octa = []

# #         for i in batch_indices:
# #             oct_triplet = [self.load_image(self.oct_images[i + j], gray=True).T for j in range(3)] # change 5 to 3 for 3N model
# #             concatenated_oct = np.concatenate(oct_triplet, axis=-1)

# #             octa_image = self.load_image(self.octa_images[i + 1], gray=True)

# #             batch_oct.append(concatenated_oct)
# #             batch_octa.append(octa_image)

# #         batch_oct = np.array(batch_oct)
# #         batch_octa = np.array(batch_octa)

# #         print(f'batch_oct shape: {batch_oct.shape}')
# #         print(f'batch_octa shape: {batch_octa.shape}')

# #         return batch_oct, batch_octa

# #     def load_image(self, path, gray=False):
# #         image = cv2.imread(path, cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
# #         image = cv2.resize(image, self.image_size)

# #         if gray:
# #             image = np.expand_dims(image, axis=-1)

# #         image = image.astype(np.float32) / 255.0
# #         return image

# # print("Loading test data directories")
# # test_subdirs = sorted([d.name for d in oct_root_path.iterdir() if d.is_dir()])

# # test_generator = SVCNetTestDataGenerator(oct_root_path, octa_root_path, batch_size=batch_size, image_size=(image_width, image_height), subdir_list=test_subdirs)

# # # Evaluate the model
# # test_loss, test_mse, test_mae = model.evaluate(test_generator)
# # print(f"Test Loss: {test_loss}, Test MSE: {test_mse}, Test MAE: {test_mae}")

# # # Predict on the test set
# # predictions = model.predict(test_generator)

# # # Save predictions
# # predictions_dir = os.path.join(export_dir, "predictions/Fingernail_retina")
# # if not os.path.exists(predictions_dir):
# #     os.makedirs(predictions_dir)

# # for i, pred in enumerate(predictions):
# #     pred_image = (pred * 255).astype(np.uint8)
# #     pred_image_path = os.path.join(predictions_dir, f"pred_{i}.png")
# #     cv2.imwrite(pred_image_path, pred_image)

# # print(f"Predictions saved to: {predictions_dir}")




# import tensorflow as tf
# import numpy as np
# import os
# import cv2
# from pathlib import Path
# from model import *
# from keras.saving import register_keras_serializable

# # Configuration
# batch_size = 4
# image_height, image_width = 640, 320

# # Define paths
# oct_root_path = Path("/scratch/at5282/Lab-PI/Fingernail/Data/Test-Data/OCT")
# octa_root_path = Path("/scratch/at5282/Lab-PI/Fingernail/Data/Test-Data/OCTA")
# export_dir = Path("/scratch/at5282/Lab-PI/New_Results")
# final_model_path = export_dir / "revised_size_final_svcnet_model.hdf5"

# @register_keras_serializable()
# def SSIMLoss(y_true, y_pred):
#     return (1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0)))

# # Load the model
# model = tf.keras.models.load_model(final_model_path, custom_objects={'SSIMLoss': SSIMLoss})

# # Generator class
# class SVCNetTestDataGenerator(tf.keras.utils.Sequence):
#     def __init__(self, oct_root_path, octa_root_path, batch_size, image_size, paired_subdirs):
#         self.oct_root_path = Path(oct_root_path)
#         self.octa_root_path = Path(octa_root_path)
#         self.batch_size = batch_size
#         self.image_size = image_size

#         self.oct_images = []
#         self.octa_images = []

#         for oct_subdir, octa_subdir in paired_subdirs:
#             oct_dir = self.oct_root_path / oct_subdir
#             octa_dir = self.octa_root_path / octa_subdir

#             oct_files = sorted(list(oct_dir.glob("*.png")))
#             octa_files = sorted(list(octa_dir.glob("*.png")))

#             min_len = min(len(oct_files), len(octa_files))

#             if min_len < 3:
#                 print(f"Skipping ({oct_subdir}, {octa_subdir}) due to insufficient images.")
#                 continue

#             self.oct_images.extend(oct_files[:min_len])
#             self.octa_images.extend(octa_files[:min_len])

#         self.indices = list(range(len(self.oct_images) - 2))

#         if len(self.indices) < self.batch_size:
#             print("Warning: Not enough data to form a full batch.")

#     def __len__(self):
#         return len(self.indices) // self.batch_size

#     def __getitem__(self, idx):
#         batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
#         batch_oct, batch_octa = [], []

#         for i in batch_indices:
#             oct_triplet = [self.load_image(self.oct_images[i + j], gray=True) for j in range(3)]
#             concatenated_oct = np.concatenate(oct_triplet, axis=-1)
#             octa_image = self.load_image(self.octa_images[i + 1], gray=True)

#             batch_oct.append(concatenated_oct)
#             batch_octa.append(octa_image)

#         return np.array(batch_oct), np.array(batch_octa)

#     def load_image(self, path, gray=False):
#         image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
#         image = cv2.resize(image, self.image_size)
#         if gray:
#             image = np.expand_dims(image, axis=-1)
#         return image.astype(np.float32) / 255.0

# # ---- Load subdirs and generate pairs ----
# print("Loading test directories...")

# oct_subdirs = sorted([d.name.strip().replace(" ", "") for d in oct_root_path.iterdir() if d.is_dir()])
# octa_subdirs = sorted([d.name.strip().replace(" ", "") for d in octa_root_path.iterdir() if d.is_dir()])

# oct_ids = sorted(set(s[:3] for s in oct_subdirs if "OCT" in s))
# octa_ids = sorted(set(s[:3] for s in octa_subdirs if "Speckle" in s))
# common_ids = sorted(set(oct_ids).intersection(octa_ids))

# def generate_pairs(ids):
#     return [
#         (f"{id}_XOCT", f"{id}_XSpeckle") for id in ids
#     ] + [
#         (f"{id}_YOCT", f"{id}_YSpeckle") for id in ids
#     ] + [
#         (f"{id}_ZOCT", f"{id}_ZSpeckle") for id in ids
#     ]

# test_pairs = generate_pairs(common_ids)

# # Create generator
# test_generator = SVCNetTestDataGenerator(
#     oct_root_path, octa_root_path,
#     batch_size=batch_size,
#     image_size=(image_width, image_height),
#     paired_subdirs=test_pairs
# )

# # ---- Evaluate ----
# test_loss, test_mse, test_mae = model.evaluate(test_generator)
# print(f"[Test Results] Loss: {test_loss}, MSE: {test_mse}, MAE: {test_mae}")

# # ---- Predict and save ----
# print("Generating predictions...")
# predictions = model.predict(test_generator)

# predictions_dir = export_dir / "predictions/Fingernail_retina"
# predictions_dir.mkdir(parents=True, exist_ok=True)

# for i, pred in enumerate(predictions):
#     pred_image = (pred * 255).astype(np.uint8)
#     pred_image = pred_image.squeeze()  # Remove channel dimension
#     pred_image_path = predictions_dir / f"pred_{i}.png"
#     cv2.imwrite(str(pred_image_path), pred_image)

# print(f"Saved {len(predictions)} prediction images to: {predictions_dir}")