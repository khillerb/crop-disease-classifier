from datasets import load_dataset, Dataset, Image
from glob import glob
import os
from PIL import Image
from collections import Counter
import random
from tqdm import tqdm
import pickle


key_dict = {
    "soy": {"bacterial blight": 0, "frogeye": 1, "healthy": 2, "soy rust": 3},
    "corn": {"leaf blight": 0, "common rust": 1, "gray leaf spot": 2, "healthy": 3},
    "rice": {
        "bacterial blight": 0,
        "blast": 1,
        "brown spot": 2,
        "healthy": 3,
        "tungro": 4,
    },
    "potato": {
        "Bacteria": 0,
        "Fungi": 1,
        "Healthy": 2,
        "Nematode": 3,
        "Pest": 4,
        "Phytopthora": 5,
        "Virus": 6,
    },
    "wheat": {
        "BlackPoint": 0,
        "FusariumFootRot": 1,
        "HealthyLeaf": 2,
        "LeafBlight": 3,
        "WheatBlast": 4,
    },
}


def hf_to_dict_with_pil(dataset):
    out = {key: [] for key in dataset.column_names}

    for ex in dataset:
        for k, v in ex.items():
            if k == "image":
                out[k].append(v.convert("RGB"))
            else:
                out[k].append(v)

    return out


# Load existing datasets
print("Loading soy, corn, and rice datasets...")
soy_ds = load_dataset("anandvermagmailcom/soybean-leaf-diseases")
corn_ds = load_dataset("Prachi1234/corn-leaf-disease")
rice_ds = load_dataset("sharmin3/Rice-Leaf-Disease")

# Process soy dataset
print("Processing soy dataset...")
soy_processed = soy_ds["train"]
soy_processed = soy_processed.add_column("plant", ["soy"] * len(soy_processed))

# Process corn dataset
print("Processing corn dataset...")
corn_processed = corn_ds["train"]
corn_processed = corn_processed.add_column("plant", ["corn"] * len(corn_processed))

# Process rice dataset
print("Processing rice dataset...")
rice_processed = rice_ds["train"]
rice_processed = rice_processed.add_column("plant", ["rice"] * len(rice_processed))

# Convert processed data to PIL dicts
soy_data = hf_to_dict_with_pil(soy_processed)
corn_data = hf_to_dict_with_pil(corn_processed)
rice_data = hf_to_dict_with_pil(rice_processed)
print(corn_data)

# Process potato dataset from folders
print("Processing potato dataset...")
potato_data = {"image": [], "label": [], "plant": []}
potato_path = "data/Potato Leaf Disease Dataset"

for disease_folder in tqdm(glob(os.path.join(potato_path, "*"))):
    if os.path.isdir(disease_folder):
        disease_name = os.path.basename(disease_folder)
        for img_path in glob(os.path.join(disease_folder, "*")):
            if img_path.lower().endswith((".png", ".jpg", ".jpeg")):
                try:
                    img = Image.open(img_path).convert("RGB")
                    potato_data["image"].append(img)
                    potato_data["label"].append(key_dict["potato"][disease_name])
                    potato_data["plant"].append("potato")
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")


# Process wheat dataset from folders
print("Processing wheat dataset...")
wheat_data = {"image": [], "label": [], "plant": []}
wheat_path = "data/Wheat Dataset"

for disease_folder in tqdm(glob(os.path.join(wheat_path, "*"))):
    if os.path.isdir(disease_folder):
        disease_name = os.path.basename(disease_folder)
        nex_path = os.path.join(disease_folder, "*")
        print(f" - Processing {nex_path}...")
        for img_path in glob(nex_path):
            if img_path.lower().endswith((".png", ".jpg", ".jpeg")):
                try:
                    img = Image.open(img_path).convert("RGB")
                    wheat_data["image"].append(img)
                    wheat_data["label"].append(key_dict["wheat"][disease_name])
                    wheat_data["plant"].append("wheat")
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")


# Function to balance disease classes within each plant dataset
def balance_disease_classes(dataset: dict, plant_name: str):
    print(f"Balancing disease classes for {plant_name}...")

    diseases = dataset["label"].copy()

    # Count diseases
    disease_counts = Counter(diseases)
    min_count = min(disease_counts.values())

    print(f"  Disease distribution before balancing: {dict(disease_counts)}")
    print(f"  Balancing to {min_count} samples per disease")

    # Group indices by disease
    disease_indices = {}
    for idx, disease in enumerate(diseases):
        disease_indices.setdefault(disease, []).append(idx)

    # Sample min_count from each disease
    balanced_indices = []
    for indices in disease_indices.values():
        random.shuffle(indices)
        balanced_indices.extend(indices[:min_count])

    random.shuffle(balanced_indices)

    # Rebuild dataset dict
    balanced_dataset = {
        key: [values[i] for i in balanced_indices] for key, values in dataset.items()
    }

    return balanced_dataset


# Balance classes in each plant's dataset
print("\nBalancing individual plant datasets...")
soy_balanced = balance_disease_classes(soy_data, "soy")
corn_balanced = balance_disease_classes(corn_data, "corn")
rice_balanced = balance_disease_classes(rice_data, "rice")
potato_balanced = balance_disease_classes(potato_data, "potato")
wheat_balanced = balance_disease_classes(wheat_data, "wheat")
print(corn_data)


# Save balanced plant datasets
def save_datadict(data, name):
    with open(f"{name}.pkl", "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


print("\nSaving individual plant datasets...")
save_datadict(soy_balanced, "soy_balanced_dataset")
save_datadict(corn_balanced, "corn_balanced_dataset")
save_datadict(rice_balanced, "rice_balanced_dataset")
save_datadict(potato_balanced, "potato_balanced_dataset")
save_datadict(wheat_balanced, "wheat_balanced_dataset")

# Concatenate all plant datasets
print("\nConcatenating all plant datasets...")
key_dict["all"] = {"soy": 0, "corn": 1, "rice": 2, "potato": 3, "wheat": 4}
labeler = lambda label: key_dict["all"][label]
all_sets = [soy_data, corn_data, rice_data, potato_data, wheat_data]
all_data = {"image": [], "label": []}
for i in range(len(all_sets)):
    all_data["image"] = all_data["image"] + all_sets[i]["image"]
    all_data["label"] = all_data["label"] + all_sets[i]["plant"]
all_data["label"] = [labeler(label) for label in all_data["label"]]

print("All data labels: ", list(set(all_data["label"])))
# Balance plant classes
print("Balancing plant classes...")
plant_counts = Counter(all_data["label"])
min_plant_count = min(plant_counts.values())

print(f"  Plant distribution before balancing: {dict(plant_counts)}")
print(f"  Balancing to {min_plant_count} samples per plant")

# Group indices by plant
plant_indices = {}
for idx, plant in enumerate(all_data["label"]):
    if plant not in plant_indices:
        plant_indices[plant] = []
    plant_indices[plant].append(idx)

# Sample min_plant_count from each plant
balanced_plant_indices = []
for plant, indices in plant_indices.items():
    random.shuffle(indices)
    balanced_plant_indices.extend(indices[:min_plant_count])

random.shuffle(balanced_plant_indices)
all_plants_balanced = {"image": [], "label": []}
for idx in balanced_plant_indices:
    all_plants_balanced["image"].append(all_data["image"][idx])
    all_plants_balanced["label"].append(all_data["label"][idx])

# Save final balanced dataset
print("\nSaving final balanced dataset...")
save_datadict(all_plants_balanced, "all_plants_balanced_dataset")
save_datadict(key_dict, "key_dict")
print("\nDataset processing complete!")
print(f"Final dataset size: {len(all_plants_balanced)} samples")
print(f"Plant distribution: {dict(Counter(all_plants_balanced['label']))}")
