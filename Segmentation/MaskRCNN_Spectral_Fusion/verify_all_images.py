import pandas as pd
import os
from pathlib import Path

csv_path = r"D:\Projects\Solar Detection\Data Analytics\EI_train_data(Sheet1).csv"
data_dir = Path(r"D:\Projects\Solar Detection\Data Analytics\Google_MapStaticAPI\images")

df = pd.read_csv(csv_path)
print(f"CSV: {len(df)} samples")
print(f"  Solar: {df['has_solar'].sum()}")
print(f"  No solar: {(df['has_solar'] == 0).sum()}")

found_count = 0
missing_samples = []

for idx, row in df.iterrows():
    sampleid = str(row['sampleid']).zfill(4)
    sample_int = int(row['sampleid'])
    has_solar = int(row['has_solar'])
    
    # Try patterns based on has_solar label
    if has_solar == 1:
        patterns = [
            f"{sample_int}.0_1.0.png",
            f"{sampleid}_1.png",
            f"{sampleid}.png",
        ]
    else:
        patterns = [
            f"{sample_int}.0_0.0.png",
            f"{sampleid}_0.png",
            f"{sampleid}.png",
        ]
    
    found = False
    for pattern in patterns:
        if (data_dir / pattern).exists():
            found = True
            break
    
    if found:
        found_count += 1
    else:
        missing_samples.append(row['sampleid'])

print(f"\n✅ FOUND: {found_count}/{len(df)} images")
print(f"❌ MISSING: {len(missing_samples)} images")

if missing_samples[:20]:
    print(f"\nMissing sample IDs (first 20): {missing_samples[:20]}")
