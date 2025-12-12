# DEPLOYMENT GUIDE - Solar Panel Detection Application

## ✅ What Has Been Created

### Complete Full-Stack Application
1. **Backend (FastAPI)** - `main.py`
   - Coordinate search with proximity matching (Haversine distance)
   - CSV data loading (3000 records)
   - JSON detection data retrieval
   - Visualization image serving
   - RESTful API with CORS support
   - Health checks and statistics endpoints

2. **Frontend (React + TypeScript)** - `frontend/src/App.tsx`
   - Modern, responsive UI with Tailwind CSS
   - Latitude/Longitude input fields
   - Search radius configuration
   - Real-time coordinate validation
   - Result display with:
     - Match status
     - Distance calculation
     - Solar panel detection status
     - Visualization image viewer
     - JSON data viewer

3. **Docker Configuration**
   - Multi-stage Dockerfile for optimized builds
   - Docker Compose for orchestration
   - Nginx reverse proxy configuration
   - Environment configuration

4. **Supporting Files**
   - README.md - Complete documentation
   - requirements.txt - Python dependencies
   - package.json - Node dependencies
   - test_api.py - API testing script

---

## 🚀 QUICK START GUIDE

### Method 1: Run Locally (Development)

#### Backend Only (FastAPI)
```powershell
# Navigate to Production folder
cd "d:\Projects\Solar Detection\Production"

# Install Python dependencies (if not already installed)
pip install fastapi uvicorn pandas pydantic

# Start the backend server
python main.py
```
**Access:** http://localhost:8000
**API Docs:** http://localhost:8000/docs

#### Frontend (React)
```powershell
# Navigate to frontend folder
cd "d:\Projects\Solar Detection\Production\frontend"

# Install dependencies (first time only)
npm install

# Start development server
npm start
```
**Access:** http://localhost:3000

---

### Method 2: Run with Docker

#### Option A: Docker Compose (Recommended)
```powershell
cd "d:\Projects\Solar Detection\Production"

# Build and start all services
docker-compose up --build

# Or run in background
docker-compose up -d --build
```

**Access Points:**
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

**Stop Services:**
```powershell
docker-compose down
```

#### Option B: Docker (Single Container)
```powershell
cd "d:\Projects\Solar Detection\Production"

# Build image
docker build -t solar-detection .

# Run container
docker run -p 8000:8000 -v "${PWD}/Output:/app/Output:ro" solar-detection
```

---

## 📋 HOW TO USE THE APPLICATION

### Step 1: Start the Application
Choose one of the methods above to start the backend and frontend.

### Step 2: Open the Frontend
Navigate to http://localhost:3000 in your web browser.

### Step 3: Enter Coordinates
Enter coordinates from your dataset. Try these examples:

**Sample Coordinates:**
- Latitude: `21.11011407`, Longitude: `72.86434589` (Sample 0001)
- Latitude: `21.5030457`, Longitude: `70.45946829` (Sample 0002)
- Latitude: `21.19117593`, Longitude: `72.79289965` (Sample 0003)

### Step 4: Set Search Radius
Default is 0.5 km. Adjust if needed (e.g., 0.1 for exact match, 5.0 for wider search).

### Step 5: Click "Check Location"
The system will:
1. Search the CSV database for matching coordinates
2. Calculate distance using Haversine formula
3. Display results if found within tolerance
4. Load visualization image (if available)
5. Load JSON detection data (if available)

### Step 6: View Results
If a match is found, you'll see:
- ✅ Match confirmation with distance
- Sample ID
- Exact coordinates from database
- Solar panel detection status (Yes/No)
- Visualization image (if exists in Output/visualizations/)
- JSON detection data (if exists in Output/json_records/)

---

## 🧪 TESTING THE API

### Manual Testing with curl
```powershell
# Health check
curl http://localhost:8000/health

# Get statistics
curl http://localhost:8000/stats

# Check coordinates
curl -X POST http://localhost:8000/check `
  -H "Content-Type: application/json" `
  -d '{"latitude": 21.1101, "longitude": 72.8643, "tolerance": 0.5}'

# Get JSON data
curl http://localhost:8000/json/0001

# List samples
curl "http://localhost:8000/list?has_solar=1&limit=10"
```

### Automated Testing
```powershell
cd "d:\Projects\Solar Detection\Production"

# Run test script (requires requests library)
pip install requests
python test_api.py
```

### Interactive API Documentation
Visit http://localhost:8000/docs for Swagger UI where you can:
- Test all endpoints interactively
- See request/response schemas
- Try different parameter values

---

## 📁 FILE STRUCTURE REFERENCE

```
Production/
├── main.py                     # FastAPI backend (renamed from fastapi.py)
├── EI_train_data(Sheet1).csv  # Training data (3000 records)
├── Output/                     # Detection results
│   ├── json_records/          # JSON files by sample ID
│   │   ├── 0001.json
│   │   ├── 0002.json
│   │   └── ...
│   └── visualizations/        # PNG images by sample ID
│       ├── 0001.png
│       ├── 0002.png
│       └── ...
├── frontend/                   # React application
│   ├── src/
│   │   ├── App.tsx           # Main component
│   │   └── index.tsx         # Entry point
│   ├── public/
│   │   └── index.html        # HTML template
│   ├── package.json
│   └── tsconfig.json
├── Dockerfile                  # Docker build configuration
├── docker-compose.yml         # Multi-container orchestration
├── nginx.conf                 # Nginx configuration
├── requirements.txt           # Python dependencies
├── test_api.py               # API testing script
├── README.md                  # Full documentation
└── DEPLOYMENT_GUIDE.md       # This file
```

---

## 🔧 TROUBLESHOOTING

### Backend Issues

**Problem:** `ImportError: cannot import name 'FastAPI'`
**Solution:** File was renamed from `fastapi.py` to `main.py` to avoid conflict.

**Problem:** `FileNotFoundError: EI_train_data(Sheet1).csv`
**Solution:** Ensure CSV file is in the Production directory.

**Problem:** Port 8000 already in use
**Solution:**
```powershell
# Find process using port 8000
netstat -ano | findstr :8000

# Kill process (replace PID)
taskkill /PID <PID> /F

# Or use different port
uvicorn main:app --port 8001
```

### Frontend Issues

**Problem:** `Module not found` errors
**Solution:**
```powershell
cd frontend
npm install
```

**Problem:** API requests failing (CORS)
**Solution:** Backend has CORS enabled for all origins. Check backend is running on port 8000.

**Problem:** Visualization not loading
**Solution:** Ensure visualization PNG files exist in `Output/visualizations/` with correct naming (e.g., `0001.png`).

### Docker Issues

**Problem:** Docker build fails
**Solution:**
```powershell
# Clean Docker cache
docker system prune -a

# Rebuild
docker-compose build --no-cache
```

**Problem:** Volume mount issues on Windows
**Solution:** Ensure Docker Desktop has access to the drive. Check Docker Desktop → Settings → Resources → File Sharing.

---

## 🔐 IMPORTANT NOTES

1. **CSV Format:** The CSV must have columns: `sampleid`, `latitude`, `longitude`, `has_solar`

2. **Sample ID Format:** JSON and visualization files must be named with 4-digit zero-padded IDs:
   - JSON: `0001.json`, `0002.json`, etc.
   - Images: `0001.png`, `0002.png`, etc.

3. **Coordinate Matching:** Uses Haversine distance for accurate great-circle distance calculation.

4. **Tolerance:** Default is 0.5 km. Adjust based on your precision needs.

5. **Performance:** The application loads all 3000 records into memory for fast lookup.

---

## 📊 API ENDPOINTS SUMMARY

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/stats` | GET | Dataset statistics |
| `/check` | POST | Check coordinates |
| `/json/{sampleid}` | GET | Get JSON data |
| `/visualization/{sampleid}` | GET | Get image |
| `/list` | GET | List samples |
| `/docs` | GET | Swagger UI |
| `/redoc` | GET | ReDoc UI |

---

## 🌐 PRODUCTION DEPLOYMENT

### On Cloud Server (AWS, Azure, GCP)

1. **Push to GitHub/GitLab**
2. **SSH into server**
3. **Clone repository**
4. **Install Docker**
5. **Run:**
   ```bash
   docker-compose up -d
   ```
6. **Configure firewall:**
   ```bash
   sudo ufw allow 80
   sudo ufw allow 443
   ```
7. **Set up domain and SSL (optional):**
   - Use Nginx or Caddy as reverse proxy
   - Get SSL certificate with Let's Encrypt

### On Local Network

1. **Find your local IP:**
   ```powershell
   ipconfig
   ```
2. **Update `REACT_APP_API_URL` in frontend:**
   ```javascript
   const API_BASE_URL = 'http://YOUR_LOCAL_IP:8000';
   ```
3. **Start services**
4. **Access from other devices on network:**
   - Frontend: `http://YOUR_LOCAL_IP:3000`
   - Backend: `http://YOUR_LOCAL_IP:8000`

---

## ✨ FEATURES SUMMARY

### Backend Features
- ✅ Coordinate proximity search
- ✅ Haversine distance calculation
- ✅ CSV data loading (3000 records)
- ✅ JSON response serving
- ✅ Image file serving
- ✅ RESTful API design
- ✅ CORS enabled
- ✅ Health checks
- ✅ Statistics endpoint
- ✅ Filtering by solar status
- ✅ Pagination support

### Frontend Features
- ✅ Modern, responsive UI
- ✅ Real-time input validation
- ✅ Loading states
- ✅ Error handling
- ✅ Coordinate input fields
- ✅ Search radius configuration
- ✅ Result visualization
- ✅ Image display
- ✅ JSON viewer
- ✅ Distance display
- ✅ Status indicators
- ✅ Keyboard shortcuts (Enter to search)

---

## 🎯 NEXT STEPS

1. **Test the application** with your actual data
2. **Add more sample visualizations** to Output/visualizations/
3. **Add JSON detection data** to Output/json_records/
4. **Customize the UI** (colors, branding, etc.)
5. **Add authentication** (if needed for production)
6. **Set up monitoring** (logging, error tracking)
7. **Deploy to production** server

---

## 📞 SUPPORT

For issues or questions:
1. Check this guide
2. Review the README.md
3. Test with `test_api.py`
4. Check API docs at `/docs`
5. Review server logs

---

**Last Updated:** December 12, 2025
**Version:** 1.0.0
**Status:** ✅ Ready for Testing & Deployment
