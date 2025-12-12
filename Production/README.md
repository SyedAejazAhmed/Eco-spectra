# Solar Panel Detection API & Frontend

A full-stack application to check if a location has solar panels installed, built with FastAPI (backend) and React (frontend), deployable with Docker.

## Features

- 🗺️ **Coordinate Search**: Enter latitude/longitude to find matching locations
- 📊 **Visual Results**: View detection visualizations if available
- 📄 **JSON Data**: Access detailed detection data in JSON format
- 🎯 **Proximity Search**: Configurable search radius to find nearby locations
- 🐳 **Docker Ready**: Complete Docker setup for easy deployment

## Project Structure

```
Production/
├── fastapi.py                  # FastAPI backend
├── EI_train_data(Sheet1).csv  # Training data with coordinates
├── Output/                     # Detection results
│   ├── json_records/          # JSON detection data
│   └── visualizations/        # Detection images
├── frontend/                   # React frontend
│   ├── src/
│   │   ├── App.tsx           # Main React component
│   │   └── index.tsx         # Entry point
│   ├── public/
│   │   └── index.html
│   ├── package.json
│   └── tsconfig.json
├── Dockerfile                  # Docker build configuration
├── docker-compose.yml         # Docker Compose orchestration
├── requirements.txt           # Python dependencies
├── nginx.conf                 # Nginx configuration
└── README.md                  # This file
```

## Quick Start

### Option 1: Docker (Recommended)

1. **Build and run with Docker Compose:**
   ```bash
   docker-compose up --build
   ```

2. **Access the application:**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

### Option 2: Local Development

#### Backend Setup

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the FastAPI server:**
   ```bash
   uvicorn fastapi:app --reload --host 0.0.0.0 --port 8000
   ```

3. **Test the API:**
   ```bash
   curl http://localhost:8000/health
   ```

#### Frontend Setup

1. **Navigate to frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Start the development server:**
   ```bash
   npm start
   ```

4. **Open in browser:**
   - http://localhost:3000

## API Endpoints

### Health Check
```
GET /health
```
Returns API health status and loaded records count.

### Check Coordinates
```
POST /check
Content-Type: application/json

{
  "latitude": 21.1101,
  "longitude": 72.8643,
  "tolerance": 0.5
}
```
Searches for locations within tolerance (km) and returns match details.

### Get JSON Data
```
GET /json/{sampleid}
```
Returns detection JSON data for a specific sample.

### Get Visualization
```
GET /visualization/{sampleid}
```
Returns detection visualization image for a specific sample.

### List Samples
```
GET /list?has_solar=1&limit=100
```
Lists available samples with optional filtering.

### Dataset Statistics
```
GET /stats
```
Returns dataset statistics (total locations, solar/no-solar counts, coordinate ranges).

## Usage Example

1. **Enter coordinates:**
   - Latitude: `21.1101`
   - Longitude: `72.8643`
   - Search Radius: `0.5` km

2. **Click "Check Location"**

3. **View results:**
   - ✅ Location found with distance
   - 🌞 Solar panel detection status
   - 🖼️ Visualization image (if available)
   - 📄 Detection JSON data (if available)

## Sample Coordinates (from dataset)

Try these coordinates:
- `21.11011407, 72.86434589` - Sample 0001 (has solar)
- `21.5030457, 70.45946829` - Sample 0002 (has solar)
- `21.19117593, 72.79289965` - Sample 0003 (has solar)

## Docker Commands

### Build the image:
```bash
docker build -t solar-detection .
```

### Run the container:
```bash
docker run -p 8000:8000 solar-detection
```

### Run with Docker Compose:
```bash
docker-compose up -d
```

### View logs:
```bash
docker-compose logs -f
```

### Stop containers:
```bash
docker-compose down
```

## Environment Variables

### Backend (FastAPI)
- `PYTHONUNBUFFERED=1` - Python output buffering

### Frontend (React)
- `REACT_APP_API_URL` - Backend API URL (default: http://localhost:8000)

## Technology Stack

### Backend
- **FastAPI** - Modern Python web framework
- **Pandas** - Data manipulation
- **Uvicorn** - ASGI server
- **Pydantic** - Data validation

### Frontend
- **React 18** - UI framework
- **TypeScript** - Type-safe JavaScript
- **Tailwind CSS** - Styling (via CDN)
- **Lucide React** - Icon library

### Deployment
- **Docker** - Containerization
- **Docker Compose** - Multi-container orchestration
- **Nginx** - Frontend web server & reverse proxy

## API Response Examples

### Successful Match
```json
{
  "found": true,
  "sampleid": "0001",
  "latitude": 21.11011407,
  "longitude": 72.86434589,
  "has_solar": 1,
  "distance_km": 0.123,
  "json_available": true,
  "visualization_available": true,
  "message": "Match found! Distance: 0.123 km. Solar panels: Yes"
}
```

### No Match
```json
{
  "found": false,
  "json_available": false,
  "visualization_available": false,
  "message": "No location found within 0.50 km of (21.5, 72.5)"
}
```

## Troubleshooting

### Port already in use
Change ports in `docker-compose.yml`:
```yaml
ports:
  - "8001:8000"  # Backend
  - "3001:80"    # Frontend
```

### CSV file not found
Ensure `EI_train_data(Sheet1).csv` is in the Production directory.

### Output directory missing
Create the Output directory structure:
```bash
mkdir -p Output/json_records Output/visualizations
```

### CORS errors
Backend CORS is configured to allow all origins. Check `REACT_APP_API_URL` if needed.

## Development

### Adding new features
1. Update `fastapi.py` for backend changes
2. Update `frontend/src/App.tsx` for UI changes
3. Rebuild Docker images: `docker-compose up --build`

### Testing
```bash
# Backend tests
curl -X POST http://localhost:8000/check \
  -H "Content-Type: application/json" \
  -d '{"latitude": 21.1101, "longitude": 72.8643, "tolerance": 0.5}'

# Frontend
npm test
```

## License

This project is part of the Solar Detection research project.

## Contact

For questions or issues, please contact the development team.
