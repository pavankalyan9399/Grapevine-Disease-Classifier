import os
import zipfile

# Define paths
dataset_zip = r"C:\PFSD\Deep Learning Skill\Skill_project\dataset\dataset.zip"
extract_to = r"C:\PFSD\Deep Learning Skill\Skill_project\dataset"

# Extract dataset if not already extracted
if not os.path.exists(os.path.join(extract_to, "Original Data", "train")):
    print("Extracting dataset...")
    with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete!")
else:
    print("Dataset already extracted.")
