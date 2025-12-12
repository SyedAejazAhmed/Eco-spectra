import pandas as pd
import os

csv_path = r"D:\Projects\Solar Detection\Data Analytics\EI_train_data(Sheet1).csv"
image_dir = r"D:\Projects\Solar Detection\Data Analytics\Google_MapStaticAPI\images"

# Load CSV
df = pd.read_csv(csv_path)
print(f"CSV: {len(df)} samples")
print(f"  Solar: {df['has_solar'].sum()}")
print(f"  No solar: {(df['has_solar'] == 0).sum()}")

# Check which samples are missing
missing_solar = []
missing_no_solar = []
found_solar = []
found_no_solar = []

patterns = [
    "{sid}.0_1.0.png",
    "{sid}_1.png",
    "{sid}.png",
    "{num}.0_1.0.png",
    "{num}_1.png"
]

for _, row in df.iterrows():
    sid = str(row['sampleid'])
    has_solar = int(row['has_solar'])
    num = int(float(sid)) if sid.replace('.', '').isdigit() else None
    
    # Try all patterns
    found = False
    for pattern in patterns:
        if "{sid}" in pattern:
            path = os.path.join(image_dir, pattern.format(sid=sid))
        else:
            if num is None:
                continue
            path = os.path.join(image_dir, pattern.format(num=num))
        
        if os.path.exists(path):
            found = True
            break
    
    if found:
        if has_solar:
            found_solar.append(sid)
        else:
            found_no_solar.append(sid)
    else:
        if has_solar:
            missing_solar.append(sid)
        else:
            missing_no_solar.append(sid)

print(f"\n✅ FOUND: {len(found_solar) + len(found_no_solar)} total")
print(f"  Solar: {len(found_solar)}")
print(f"  No solar: {len(found_no_solar)}")

print(f"\n❌ MISSING: {len(missing_solar) + len(missing_no_solar)} total")
print(f"  Solar: {len(missing_solar)}")
print(f"  No solar: {len(missing_no_solar)}")

if missing_no_solar:
    print(f"\n⚠️ Missing NO-SOLAR samples (first 20): {missing_no_solar[:20]}")
if missing_solar:
    print(f"⚠️ Missing SOLAR samples (first 20): {missing_solar[:20]}")
