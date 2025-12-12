import requests
import json

API_BASE_URL = "http://localhost:8000"

def test_coordinates(lat, lon):
    """Test coordinate lookup"""
    print(f"\n{'='*60}")
    print(f"Testing coordinates: {lat}, {lon}")
    print('='*60)
    
    # Check coordinates
    response = requests.post(
        f"{API_BASE_URL}/check",
        json={"latitude": lat, "longitude": lon}
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n✓ Response Status: {response.status_code}")
        print(f"\nResult:")
        print(f"  Found: {data['found']}")
        
        if data['found']:
            print(f"  Sample ID: {data['sampleid']}")
            print(f"  Matched Coordinates: {data['latitude']}, {data['longitude']}")
            print(f"  Distance: {data['distance_km']} km")
            print(f"  Search Radius: {data['search_radius_km']} km")
            print(f"  Solar Panels: {'Yes' if data['has_solar'] == 1 else 'No'}")
            print(f"  JSON Available: {data['json_available']}")
            print(f"  Visualization Available: {data['visualization_available']}")
            print(f"\n  Message: {data['message']}")
            
            # Try to fetch JSON if available
            if data['json_available']:
                json_response = requests.get(f"{API_BASE_URL}/json/{data['sampleid']}")
                if json_response.status_code == 200:
                    print(f"\n✓ JSON Data fetched successfully!")
                else:
                    print(f"\n✗ JSON fetch failed: {json_response.status_code}")
            
            # Check visualization
            if data['visualization_available']:
                viz_response = requests.get(f"{API_BASE_URL}/visualization/{data['sampleid']}")
                if viz_response.status_code == 200:
                    print(f"✓ Visualization image fetched successfully!")
                else:
                    print(f"✗ Visualization fetch failed: {viz_response.status_code}")
        else:
            print(f"\n  Message: {data['message']}")
    else:
        print(f"✗ Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    # Test cases
    print("\n" + "="*60)
    print("SOLAR PANEL COORDINATE TEST")
    print("="*60)
    
    # Test 1: Approximate coordinates (21.11, 72.86)
    test_coordinates(21.11, 72.86)
    
    # Test 2: More approximate coordinates
    test_coordinates(21.5, 70.46)
    
    # Test 3: Exact coordinates from dataset
    test_coordinates(21.11011407, 72.86434589)
    
    # Test 4: Non-existent location
    test_coordinates(25.0, 75.0)
    
    print("\n" + "="*60)
    print("Testing complete!")
    print("="*60 + "\n")
