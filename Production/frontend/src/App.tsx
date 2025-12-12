import React, { useState } from 'react';
import { MapPin, Search, Loader2, CheckCircle, XCircle, Image, FileJson } from 'lucide-react';

interface CoordinateResponse {
  found: boolean;
  sampleid?: string;
  latitude?: number;
  longitude?: number;
  has_solar?: number;
  distance_km?: number;
  search_radius_km?: number;
  json_available: boolean;
  visualization_available: boolean;
  message: string;
}

interface JsonData {
  [key: string]: any;
}

const SolarPanelChecker: React.FC = () => {
  const [latitude, setLatitude] = useState<string>('');
  const [longitude, setLongitude] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<CoordinateResponse | null>(null);
  const [jsonData, setJsonData] = useState<JsonData | null>(null);
  const [visualizationUrl, setVisualizationUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Use same origin for API calls (works for both Docker and dev mode)
  const API_BASE_URL = process.env.REACT_APP_API_URL || window.location.origin;

  const handleCheck = async () => {
    if (!latitude || !longitude) {
      setError('Please enter both latitude and longitude');
      return;
    }

    const lat = parseFloat(latitude);
    const lon = parseFloat(longitude);

    if (isNaN(lat) || isNaN(lon)) {
      setError('Invalid coordinates. Please enter valid numbers.');
      return;
    }

    if (lat < -90 || lat > 90 || lon < -180 || lon > 180) {
      setError('Coordinates out of range. Latitude: -90 to 90, Longitude: -180 to 180');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);
    setJsonData(null);
    setVisualizationUrl(null);

    try {
      // Check coordinates
      console.log('Calling API:', `${API_BASE_URL}/check`);
      const response = await fetch(`${API_BASE_URL}/check`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          latitude: lat,
          longitude: lon,
        }),
      });

      console.log('Response status:', response.status);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data: CoordinateResponse = await response.json();
      setResult(data);

      // If found, fetch JSON and visualization
      if (data.found && data.sampleid) {
        // Fetch JSON
        if (data.json_available) {
          try {
            const jsonResponse = await fetch(`${API_BASE_URL}/json/${data.sampleid}`);
            if (jsonResponse.ok) {
              const json = await jsonResponse.json();
              setJsonData(json);
            }
          } catch (e) {
            console.error('Error fetching JSON:', e);
          }
        }

        // Set visualization URL
        if (data.visualization_available) {
          setVisualizationUrl(`${API_BASE_URL}/visualization/${data.sampleid}`);
        }
      }
    } catch (err) {
      console.error('Fetch error:', err);
      const errorMessage = err instanceof Error ? err.message : 'An error occurred';
      setError(`Failed to fetch: ${errorMessage}. API URL: ${API_BASE_URL}`);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleCheck();
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex justify-center items-center mb-4">
            <MapPin className="h-12 w-12 text-indigo-600 mr-2" />
            <h1 className="text-4xl font-bold text-gray-900">Solar Panel Detection</h1>
          </div>
          <p className="text-lg text-gray-600">Check if a location has solar panels installed</p>
        </div>

        {/* Input Card */}
        <div className="bg-white rounded-lg shadow-xl p-8 mb-8">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Latitude
              </label>
              <input
                type="text"
                value={latitude}
                onChange={(e) => setLatitude(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="e.g., 21.11 or 21.11011407"
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Longitude
              </label>
              <input
                type="text"
                value={longitude}
                onChange={(e) => setLongitude(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="e.g., 72.86 or 72.86434589"
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
              />
            </div>
          </div>

          <button
            onClick={handleCheck}
            disabled={loading}
            className="w-full bg-indigo-600 text-white py-3 px-6 rounded-lg font-semibold hover:bg-indigo-700 transition duration-200 flex items-center justify-center disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? (
              <>
                <Loader2 className="animate-spin h-5 w-5 mr-2" />
                Searching...
              </>
            ) : (
              <>
                <Search className="h-5 w-5 mr-2" />
                Check Location
              </>
            )}
          </button>
        </div>

        {/* Error Display */}
        {error && (
          <div className="bg-red-50 border-l-4 border-red-500 p-4 mb-8">
            <div className="flex items-center">
              <XCircle className="h-5 w-5 text-red-500 mr-2" />
              <p className="text-red-700">{error}</p>
            </div>
          </div>
        )}

        {/* Result Display */}
        {result && (
          <div className="space-y-8">
            {/* Status Card */}
            <div
              className={`rounded-lg shadow-lg p-6 ${
                result.found
                  ? 'bg-green-50 border-l-4 border-green-500'
                  : 'bg-yellow-50 border-l-4 border-yellow-500'
              }`}
            >
              <div className="flex items-start">
                {result.found ? (
                  <CheckCircle className="h-6 w-6 text-green-500 mr-3 mt-1" />
                ) : (
                  <XCircle className="h-6 w-6 text-yellow-500 mr-3 mt-1" />
                )}
                <div className="flex-1">
                  <h3 className="text-lg font-semibold text-gray-900 mb-2">
                    {result.found ? 'Location Found!' : 'No Match Found'}
                  </h3>
                  <p className="text-gray-700 mb-4">{result.message}</p>

                  {result.found && (
                    <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                      <div>
                        <p className="text-sm text-gray-600">Sample ID</p>
                        <p className="font-semibold text-gray-900">{result.sampleid}</p>
                      </div>
                      <div>
                        <p className="text-sm text-gray-600">Coordinates</p>
                        <p className="font-semibold text-gray-900">
                          {result.latitude?.toFixed(6)}, {result.longitude?.toFixed(6)}
                        </p>
                      </div>
                      <div>
                        <p className="text-sm text-gray-600">Distance</p>
                        <p className="font-semibold text-gray-900">{result.distance_km} km</p>
                      </div>
                      <div>
                        <p className="text-sm text-gray-600">Search Radius</p>
                        <p className="font-semibold text-indigo-600">{result.search_radius_km} km</p>
                      </div>
                      <div>
                        <p className="text-sm text-gray-600">Solar Panels</p>
                        <p
                          className={`font-semibold ${
                            result.has_solar === 1 ? 'text-green-600' : 'text-gray-600'
                          }`}
                        >
                          {result.has_solar === 1 ? 'Detected ✓' : 'None'}
                        </p>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Visualization */}
            {visualizationUrl && (
              <div className="bg-white rounded-lg shadow-lg p-6">
                <div className="flex items-center mb-4">
                  <Image className="h-6 w-6 text-indigo-600 mr-2" />
                  <h3 className="text-xl font-semibold text-gray-900">Detection Visualization</h3>
                </div>
                <div className="border-2 border-gray-200 rounded-lg overflow-hidden">
                  <img
                    src={visualizationUrl}
                    alt="Solar panel detection visualization"
                    className="w-full h-auto"
                    onError={(e) => {
                      e.currentTarget.src = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="100" height="100"%3E%3Ctext x="50%25" y="50%25" text-anchor="middle" fill="%23999"%3EImage not available%3C/text%3E%3C/svg%3E';
                    }}
                  />
                </div>
              </div>
            )}

            {/* JSON Data */}
            {jsonData && (
              <div className="bg-white rounded-lg shadow-lg p-6">
                <div className="flex items-center mb-4">
                  <FileJson className="h-6 w-6 text-indigo-600 mr-2" />
                  <h3 className="text-xl font-semibold text-gray-900">Detection Data (JSON)</h3>
                </div>
                <div className="bg-gray-50 rounded-lg p-4 overflow-auto max-h-96">
                  <pre className="text-sm text-gray-800 font-mono">
                    {JSON.stringify(jsonData, null, 2)}
                  </pre>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Footer */}
        <div className="mt-12 text-center text-gray-600 text-sm">
          <p>Enter coordinates to check for solar panel installations in the database</p>
          <p className="mt-2">@Ecospectra - 2025. All rights reserved</p>
        </div>
      </div>
    </div>
  );
};

export default SolarPanelChecker;
