
from pathlib import Path
import pandas as pd

# ==========================================
# 1. PROJECT PATHS
# ==========================================
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
IMAGES = DATA / "images"
EXCEL = DATA / "Acrylamide_dataset_with_image_names.xlsx"
OUTPUT = DATA / "labels.csv"

# ==========================================
# 2. CHECK REQUIRED FILES
# ==========================================
if not EXCEL.exists():
    raise FileNotFoundError(f"Excel file not found: {EXCEL}")

if not IMAGES.exists():
    raise FileNotFoundError(f"Images folder not found: {IMAGES}")

# Find image files, including files in subfolders.
image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

image_paths = [
    path for path in IMAGES.rglob("*")
    if path.is_file() and path.suffix.lower() in image_extensions
]

# Match filenames without regard to letter case.
image_lookup = {}

for path in image_paths:
    image_lookup.setdefault(path.name.casefold(), []).append(path)

print(f"Images found on disk: {len(image_paths)}")

# ==========================================
# 3. READ EXCEL WORKBOOK
# ==========================================
sheets = pd.read_excel(EXCEL, sheet_name=None)
records = []

def add_records(sheet_name, image_column, value_column, band_column=None):
    """Collect image names and measured acrylamide values."""

    if sheet_name not in sheets:
        print(f"Warning: sheet not found: {sheet_name}")
        return

    df = sheets[sheet_name]

    if image_column not in df.columns:
        print(f"Warning: column not found: {image_column}")
        return

    if value_column not in df.columns:
        print(f"Warning: column not found: {value_column}")
        return

    for _, row in df.iterrows():
        image_name = row.get(image_column)
        value = pd.to_numeric(row.get(value_column), errors="coerce")

        if pd.isna(image_name) or pd.isna(value):
            continue

        image_name = Path(str(image_name).strip()).name

        if not image_name or image_name.lower() == "nan":
            continue

        band = None
        if band_column and band_column in df.columns:
            raw_band = row.get(band_column)
            if pd.notna(raw_band):
                band = str(raw_band).strip()

        records.append({
            "image_name": image_name,
            "acrylamide_ug_per_kg": float(value),
            "excel_band": band,
            "source_sheet": sheet_name
        })

# ==========================================
# 4. COLLECT RECORDS FROM EACH SHEET
# ==========================================

# Chapter 3: one image and one measured value.
add_records(
    "Chapter 3",
    "Image file name",
    "Acrylamide in sample (µg/kg)",
    "Acrylamide band"
)

# Chapter 4: two trials, each with its own measured value.
add_records(
    "Chapter 4",
    "Image file name (1st trial)",
    "Acrylamide in sample (µg/kg)",
    "Acrylamide band (1st trial)"
)

add_records(
    "Chapter 4",
    "Image file name (2nd trial)",
    "Acrylamide in sample (µg/kg).1",
    "Acrylamide band (2nd trial)"
)

# Chapter 5: one image and one measured value.
add_records(
    "Chapter 5",
    "Image file name",
    "Acrylamide in sample (µg/kg)",
    "Acrylamide band"
)

print(f"Excel records collected: {len(records)}")

# ==========================================
# 5. MATCH RECORDS TO IMAGE FILES
# ==========================================
matched = []
missing = []

for record in records:
    key = record["image_name"].casefold()
    candidates = image_lookup.get(key, [])

    if not candidates:
        missing.append(record)
        continue

    if len(candidates) > 1:
        print(
            f"Warning: duplicate filename on disk: "
            f"{record['image_name']}"
        )

    matched_record = record.copy()
    matched_record["image_path"] = str(candidates[0].relative_to(ROOT))
    matched.append(matched_record)

print(f"Records matched to images: {len(matched)}")
print(f"Records without matching images: {len(missing)}")

# ==========================================
# 6. CHECK FOR CONFLICTING LABELS
# ==========================================
matched_df = pd.DataFrame(matched)

if matched_df.empty:
    raise ValueError(
        "No Excel records matched your images. "
        "Check the image filenames and Excel workbook."
    )

matched_df["image_key"] = matched_df["image_name"].str.casefold()

conflicts = (
    matched_df.groupby("image_key")["acrylamide_ug_per_kg"]
    .nunique()
)

conflicting_names = conflicts[conflicts > 1].index.tolist()

if conflicting_names:
    print(
        f"Warning: {len(conflicting_names)} image filenames have "
        "different measured values in the Excel records."
    )
    print("Examples:", conflicting_names[:10])

# Keep one record per image only when the measured values agree.
# Conflicting records are excluded until their labels are checked.
matched_df = matched_df[
    ~matched_df["image_key"].isin(conflicting_names)
].copy()

matched_df = matched_df.drop_duplicates(subset="image_key", keep="first")

# ==========================================
# 7. ASSIGN PROJECT GRADES
# ==========================================
def assign_grade(value):
    if value < 500:
        return "Low"
    elif value < 1000:
        return "Medium"
    elif value < 2000:
        return "High"
    else:
        return "Very High"

matched_df["grade"] = matched_df["acrylamide_ug_per_kg"].apply(
    assign_grade
)

# ==========================================
# 8. SAVE DATASET LABELS
# ==========================================
columns = [
    "image_name",
    "image_path",
    "acrylamide_ug_per_kg",
    "grade",
    "excel_band",
    "source_sheet"
]

matched_df[columns].to_csv(OUTPUT, index=False)

print("\nGrade counts:")
print(matched_df["grade"].value_counts())

print(f"\nLabels saved to: {OUTPUT}")
print("\nDataset preparation complete.")

# Show missing filenames so you can investigate them later.
if missing:
    missing_df = pd.DataFrame(missing)
    missing_output = DATA / "unmatched_records.csv"
    missing_df.to_csv(missing_output, index=False)
    print(f"Unmatched records saved to: {missing_output}")
