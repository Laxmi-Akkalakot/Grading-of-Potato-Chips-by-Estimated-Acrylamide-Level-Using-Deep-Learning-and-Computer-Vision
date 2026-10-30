
import os
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# 1. Configuration
# ==========================================

IMAGE_SIZE = 160
BATCH_SIZE = 16
EPOCHS = 40
RANDOM_SEED = 42

tf.keras.utils.set_random_seed(RANDOM_SEED)

MODEL_PATH = "models/acrylamide_grade_v3.keras"
OUTPUT_DIR = "models/regression"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. Load dataset splits
# ==========================================

train_df = pd.read_csv("data/regression_train_split.csv")
val_df = pd.read_csv("data/regression_validation_split.csv")
test_df = pd.read_csv("data/regression_test_split.csv")

print("Training images:", len(train_df))
print("Validation images:", len(val_df))
print("Testing images:", len(test_df))

# ==========================================
# 3. Image loading function
# ==========================================

def load_images(dataframe):
    images = []

    for path in dataframe["image_path"]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")

        img = image.load_img(
            path,
            target_size=(IMAGE_SIZE, IMAGE_SIZE)
        )

        img_array = image.img_to_array(img)
        images.append(img_array)

    X = np.array(images, dtype=np.float32)
    X = preprocess_input(X)

    return X


print("\nLoading training images...")
X_train = load_images(train_df)

print("Loading validation images...")
X_val = load_images(val_df)

print("Loading testing images...")
X_test = load_images(test_df)

# ==========================================
# 4. Prepare numerical targets
# ==========================================

# Log transformation helps handle highly skewed values.
y_train = np.log1p(train_df["acrylamide_ug_per_kg"].values)
y_val = np.log1p(val_df["acrylamide_ug_per_kg"].values)
y_test = test_df["acrylamide_ug_per_kg"].values

print("\nImage shapes:")
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)

# ==========================================
# 5. Load pretrained V3 model
# ==========================================

print("\nLoading V3 classification model...")

classification_model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

backbone = classification_model.get_layer(
    "mobilenetv2_1.00_160"
)

backbone.trainable = False

print("MobileNetV2 backbone loaded successfully.")
print("Backbone frozen for initial training.")

# ==========================================
# 6. Build regression model
# ==========================================

inputs = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

# Reuse pretrained MobileNetV2 features.
x = backbone(inputs, training=False)

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)

x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.2)(x)

# One numerical output
outputs = layers.Dense(1, activation="linear")(x)

regression_model = models.Model(
    inputs=inputs,
    outputs=outputs
)

regression_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="mse",
    metrics=["mae"]
)

regression_model.summary()

# ==========================================
# 7. Training callbacks
# ==========================================

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(
            OUTPUT_DIR,
            "best_regression_model.keras"
        ),
        monitor="val_loss",
        save_best_only=True
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        min_lr=1e-7
    )
]

# ==========================================
# 8. Train regression model
# ==========================================

print("\nStarting regression model training...")

history = regression_model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    shuffle=True
)

# ==========================================
# 9. Save final model
# ==========================================

regression_model.save(
    os.path.join(OUTPUT_DIR, "acrylamide_regression.keras")
)

print("\nRegression model training completed!")

# ==========================================
# 10. Evaluate test dataset
# ==========================================

print("\nEvaluating test dataset...")

best_model = tf.keras.models.load_model(
    os.path.join(OUTPUT_DIR, "best_regression_model.keras"),
    compile=False
)

predictions_log = best_model.predict(X_test).flatten()

# Convert predictions back to original units.
predictions = np.expm1(predictions_log)

# Prevent negative concentration predictions.
predictions = np.maximum(predictions, 0)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\n========== TEST RESULTS ==========")
print(f"MAE:  {mae:.2f} µg/kg")
print(f"RMSE: {rmse:.2f} µg/kg")
print(f"R²:   {r2:.4f}")

# ==========================================
# 11. Save predictions
# ==========================================

results = test_df[[
    "image_name",
    "grade",
    "acrylamide_ug_per_kg"
]].copy()

results["predicted_acrylamide_ug_per_kg"] = predictions

results["absolute_error"] = abs(
    results["acrylamide_ug_per_kg"]
    - results["predicted_acrylamide_ug_per_kg"]
)

results.to_csv(
    "evaluation/regression_test_predictions.csv",
    index=False
)

print("\nTest predictions saved:")
print("evaluation/regression_test_predictions.csv")

print("\nTraining process finished!")
