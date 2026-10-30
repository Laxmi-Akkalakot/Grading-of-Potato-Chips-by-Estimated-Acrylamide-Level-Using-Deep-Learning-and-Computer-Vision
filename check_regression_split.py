
import pandas as pd

# Load existing splits
train = pd.read_csv("data/mobilenet_train_split.csv")
val = pd.read_csv("data/mobilenet_validation_split.csv")
test = pd.read_csv("data/mobilenet_test_split.csv")

# Display split sizes
print("TRAIN:", len(train))
print("VALIDATION:", len(val))
print("TEST:", len(test))

# Check image overlap
train_names = set(train["image_name"])
val_names = set(val["image_name"])
test_names = set(test["image_name"])

print("\nTrain-validation overlap:", len(train_names & val_names))
print("Train-test overlap:", len(train_names & test_names))
print("Validation-test overlap:", len(val_names & test_names))

# Check grade distribution
for name, df in [
    ("TRAIN", train),
    ("VALIDATION", val),
    ("TEST", test)
]:
    print(f"\n{name} grade distribution:")
    print(df["grade"].value_counts())
