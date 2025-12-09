import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf

# Define the custom loss function
def SSIMLoss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

# Load the training history
history_file_path = "/scratch/netid/Lab-PI/Results/training_history_lr_scheduled.npy"
history = np.load(history_file_path, allow_pickle=True).item()

# Extract data from history
loss = history['loss']
val_loss = history['val_loss']
mse = history['mse']
val_mse = history['val_mse']
mae = history['mae']
val_mae = history['val_mae']

# Generate epochs array
epochs = range(1, len(loss) + 1)

# Create a directory to save the plots if it doesn't exist
plot_dir = "/scratch/netid/Lab-PI/Results/plots_lr_scheduled"
os.makedirs(plot_dir, exist_ok=True)

# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(plot_dir, "loss_plot.png"))

# Plot training and validation Mean Squared Error
plt.figure(figsize=(10, 5))
plt.plot(epochs, mse, 'b', label='Training MSE')
plt.plot(epochs, val_mse, 'r', label='Validation MSE')
plt.title('Training and Validation Mean Squared Error')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.savefig(os.path.join(plot_dir, "mse_plot.png"))

# Plot training and validation Mean Absolute Error
plt.figure(figsize=(10, 5))
plt.plot(epochs, mae, 'b', label='Training MAE')
plt.plot(epochs, val_mae, 'r', label='Validation MAE')
plt.title('Training and Validation Mean Absolute Error')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.savefig(os.path.join(plot_dir, "mae_plot.png"))

# Show plots
plt.show()

