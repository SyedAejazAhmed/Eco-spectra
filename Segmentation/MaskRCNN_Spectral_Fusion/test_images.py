import pandas as pd
from pathlib import Path

csv_path = Path("D:/Projects/Solar Detection/Data Analytics/EI_train_data(Sheet1).csv")
data_dir = Path("D:/Projects/Solar Detection/Data Analytics/Google_MapStaticAPI/images")

df = pd.read_csv(csv_path)
print(f"CSV has {len(df)} samples")

found = 0
missing = []
pattern_counts = {"solar": 0, "no_solar": 0}

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
    
    img_found = False
    matched_path = None
    for pattern in patterns:
        img_path = data_dir / pattern
        if img_path.exists():
            img_found = True
            matched_path = img_path
            if has_solar == 1:
                pattern_counts["solar"] += 1
            else:
                pattern_counts["no_solar"] += 1
            break
    
    if img_found:
        found += 1
    else:
        missing.append(row['sampleid'])
    
    if idx < 10:
        status = f"✓ {matched_path.name}" if img_found else "✗ NOT FOUND"
        print(f"Sample {row['sampleid']} (has_solar={has_solar}): {status}")

print(f"\nFound: {found}/{len(df)}")
print(f"  Solar images: {pattern_counts['solar']}")
print(f"  No-solar images: {pattern_counts['no_solar']}")
print(f"Missing: {len(missing)}")

if missing:
    print(f"\nFirst 10 missing samples: {missing[:10]}")
