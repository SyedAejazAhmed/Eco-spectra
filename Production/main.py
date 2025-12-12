from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import json
from pathlib import Path
from typing import Optional
import math
import os

app = FastAPI(title="Solar Panel Detection API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load CSV data
CSV_PATH = Path("EI_train_data(Sheet1).csv")
OUTPUT_DIR = Path("Output")
JSON_DIR = OUTPUT_DIR / "json_records"
VIZ_DIR = OUTPUT_DIR / "visualizations"
STATIC_DIR = Path("static")

# Mount static files for frontend (if directory exists)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR / "static")), name="static")

# Load the training data
try:
    df = pd.read_csv(CSV_PATH)
    print(f"✓ Loaded {len(df)} records from CSV")
except Exception as e:
    print(f"Error loading CSV: {e}")
    df = pd.DataFrame()


class CoordinateRequest(BaseModel):
    latitude: float
    longitude: float


class CoordinateResponse(BaseModel):
    found: bool
    sampleid: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    has_solar: Optional[int] = None
    distance_km: Optional[float] = None
    search_radius_km: Optional[float] = None
    json_available: bool = False
    visualization_available: bool = False
    message: str


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    Returns distance in kilometers
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r


def find_closest_location(lat: float, lon: float, tolerance_km: float = 0.5):
    """
    Find the closest location in the dataset within tolerance
    """
    if df.empty:
        return None
    
    min_distance = float('inf')
    closest_record = None
    
    for idx, row in df.iterrows():
        distance = haversine_distance(lat, lon, row['latitude'], row['longitude'])
        if distance < min_distance:
            min_distance = distance
            closest_record = {
                'sampleid': row['sampleid'],
                'latitude': row['latitude'],
                'longitude': row['longitude'],
                'has_solar': row['has_solar'],
                'distance_km': distance
            }
    
    # Only return if within tolerance
    if closest_record and closest_record['distance_km'] <= tolerance_km:
        return closest_record
    
    return None


@app.get("/api")
async def api_root():
    return {
        "message": "Solar Panel Detection API",
        "total_records": len(df),
        "endpoints": [
            "/check - POST latitude & longitude to check for solar panels",
            "/health - GET health status",
            "/stats - GET dataset statistics"
        ]
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "records_loaded": len(df),
        "output_dir_exists": OUTPUT_DIR.exists()
    }


@app.get("/stats")
async def get_stats():
    if df.empty:
        raise HTTPException(status_code=500, detail="Dataset not loaded")
    
    return {
        "total_locations": len(df),
        "with_solar": int(df['has_solar'].sum()),
        "without_solar": int((df['has_solar'] == 0).sum()),
        "latitude_range": {
            "min": float(df['latitude'].min()),
            "max": float(df['latitude'].max())
        },
        "longitude_range": {
            "min": float(df['longitude'].min()),
            "max": float(df['longitude'].max())
        }
    }


@app.post("/check", response_model=CoordinateResponse)
async def check_coordinates(request: CoordinateRequest):
    """
    Check if coordinates exist in the dataset and return associated data
    """
    if df.empty:
        raise HTTPException(status_code=500, detail="Dataset not loaded")
    
    # Fixed search radius of 0.5 km
    tolerance_km = 0.5
    
    # Find closest location
    result = find_closest_location(request.latitude, request.longitude, tolerance_km)
    
    if not result:
        return CoordinateResponse(
            found=False,
            search_radius_km=tolerance_km,
            message=f"No location found within {tolerance_km} km of ({request.latitude}, {request.longitude})"
        )
    
    # Format sampleid with zero-padding (convert to int then format as 4-digit string)
    sampleid_raw = result['sampleid']
    try:
        # Handle both string and numeric sampleid
        sampleid_num = int(float(str(sampleid_raw)))
        sampleid = f"{sampleid_num:04d}"
    except:
        sampleid = str(sampleid_raw)
    
    # Check if files exist with zero-padded sampleid
    json_path = JSON_DIR / f"{sampleid}.json"
    # Try multiple visualization filename patterns
    viz_path = VIZ_DIR / f"{sampleid}_finetuned.png"
    if not viz_path.exists():
        viz_path = VIZ_DIR / f"{sampleid}.png"
    
    json_exists = json_path.exists()
    viz_exists = viz_path.exists()
    
    return CoordinateResponse(
        found=True,
        sampleid=sampleid,
        latitude=result['latitude'],
        longitude=result['longitude'],
        has_solar=result['has_solar'],
        distance_km=round(result['distance_km'], 3),
        search_radius_km=tolerance_km,
        json_available=json_exists,
        visualization_available=viz_exists,
        message=f"Match found! Distance: {result['distance_km']:.3f} km. Solar panels: {'Yes' if result['has_solar'] == 1 else 'No'}"
    )


@app.get("/json/{sampleid}")
async def get_json_data(sampleid: str):
    """
    Retrieve JSON detection results for a specific sample
    """
    json_path = JSON_DIR / f"{sampleid}.json"
    
    if not json_path.exists():
        raise HTTPException(status_code=404, detail=f"JSON file not found for sample {sampleid}")
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return JSONResponse(content=data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading JSON: {str(e)}")


@app.get("/visualization/{sampleid}")
async def get_visualization(sampleid: str):
    """
    Retrieve visualization image for a specific sample
    """
    # Try multiple filename patterns
    viz_path = VIZ_DIR / f"{sampleid}_finetuned.png"
    if not viz_path.exists():
        viz_path = VIZ_DIR / f"{sampleid}.png"
    
    if not viz_path.exists():
        raise HTTPException(status_code=404, detail=f"Visualization not found for sample {sampleid}")
    
    return FileResponse(viz_path, media_type="image/png")


@app.get("/list")
async def list_samples(has_solar: Optional[int] = None, limit: int = 100):
    """
    List available samples
    """
    if df.empty:
        raise HTTPException(status_code=500, detail="Dataset not loaded")
    
    filtered_df = df.copy()
    if has_solar is not None:
        filtered_df = filtered_df[filtered_df['has_solar'] == has_solar]
    
    samples = filtered_df.head(limit).to_dict('records')
    
    return {
        "total": len(filtered_df),
        "returned": len(samples),
        "samples": samples
    }


# Catch-all route for serving React frontend (must be last)
@app.get("/")
@app.get("/{full_path:path}")
async def serve_frontend(full_path: str = ""):
    """Serve the React frontend for all non-API routes"""
    # Try to serve the requested file
    if full_path and (STATIC_DIR / full_path).exists():
        return FileResponse(STATIC_DIR / full_path)
    
    # Otherwise serve index.html (for React Router)
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    
    return {"message": "Solar Panel Detection API - Frontend not built"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)