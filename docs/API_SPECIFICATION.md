# Jolpica-F1 API Specification & Data Pipeline

The live data pipeline queries the **Jolpica-F1 API** (the official community successor to the Ergast API).

## Endpoints

### 1. Season Schedule
- **URL:** `GET http://api.jolpi.ca/ergast/f1/{season}.json`
- **Description:** Retrieves all Grand Prix events for a target season.

### 2. Race Results
- **URL:** `GET http://api.jolpi.ca/ergast/f1/{season}/{round}/results.json`
- **Description:** Returns grid position, finishing position, points, laps, and status codes per driver.

## Rate Limits
- 200 requests / hour (unauthenticated)
- Handled gracefully in `src/api_client.py` with exponential backoff retries.
