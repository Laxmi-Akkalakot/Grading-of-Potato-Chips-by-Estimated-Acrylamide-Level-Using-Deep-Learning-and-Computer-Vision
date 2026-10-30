
import tensorflow as tf

model_path = "models/acrylamide_grade_v3.keras"

print("Loading V3 model...")

model = tf.keras.models.load_model(model_path)

print("\nMODEL SUMMARY")
model.summary()

print("\nINPUT SHAPE:")
print(model.input_shape)

print("\nOUTPUT SHAPE:")
print(model.output_shape)

print("\nMODEL LAYERS:")

for i, layer in enumerate(model.layers):
    print(i, layer.name, layer.output.shape)

print("\nPREPROCESSING LAYER DETAILS:")

preprocessing_layer = model.layers[1]

preprocessing_layer.summary()

print("\nPREPROCESSING SUB-LAYERS:")

for layer in preprocessing_layer.layers:
    print(
        "Name:", layer.name,
        "| Type:", type(layer).__name__,
        "| Config:", layer.get_config()
    )

