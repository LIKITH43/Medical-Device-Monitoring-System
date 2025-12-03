import pandas as pd

# 1. Load your existing synthetic dataset
df = pd.read_csv("SynDataset.csv")

# 2. Check that the Location column exists
if "Location" not in df.columns:
    raise ValueError("The dataset has no 'Location' column. Please check your CSV.")

# 3. Define hospitals & regions
base_hospitals = ["Hospital A", "Hospital B", "Hospital C"]   # already in your data
new_hospitals  = ["Hospital D", "Hospital E", "Hospital F"]   # we want to add
regions = ["Central Region", "East Region", "North Region", "South Region", "West Region"]

# 4. Build a mapping: copy patterns from A/B/C to D/E/F
#    e.g. rows with "Hospital A - Central Region" -> copy as "Hospital D - Central Region", etc.
pairs = [
    ("Hospital A", "Hospital D"),
    ("Hospital B", "Hospital E"),
    ("Hospital C", "Hospital F"),
]

augmented_rows = []

for src_hospital, new_hospital in pairs:
    for region in regions:
        src_loc = f"{src_hospital} - {region}"
        new_loc = f"{new_hospital} - {region}"

        # Take all rows from that source hospital+region
        subset = df[df["Location"] == src_loc].copy()
        if subset.empty:
            # If there are no rows for this combination, just skip
            continue

        # Change the Location to the new hospital+region
        subset["Location"] = new_loc
        augmented_rows.append(subset)

# 5. Concatenate everything
if augmented_rows:
    df_new = pd.concat([df] + augmented_rows, ignore_index=True)
else:
    raise RuntimeError("No matching rows were found to duplicate for new hospitals. "
                       "Check your existing 'Location' values!")

# 6. Save a new CSV (keep the original safe)
df_new.to_csv("SynDataset_upscaled.csv", index=False)

print("✅ Created SynDataset_upscaled.csv with extra hospitals D, E, and F.")
print("Old rows:", len(df), "| New rows:", len(df_new))
