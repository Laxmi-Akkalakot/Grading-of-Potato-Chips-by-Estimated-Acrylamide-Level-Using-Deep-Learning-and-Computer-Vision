
import tensorflow as tf

model = tf.keras.models.load_model(
    "models/acrylamide_grade_v3.keras"
)

backbone = model.get_layer("mobilenetv2_1.00_160")

print("Backbone name:", backbone.name)
print("Input shape:", backbone.input_shape)
print("Output shape:", backbone.output_shape)

print("\nBackbone summary:")
backbone.summary()

print("\nTrainable layers:", sum(
    1 for layer in backbone.layers if layer.trainable
))

print("Total layers:", len(backbone.layers))
