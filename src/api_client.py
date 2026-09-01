"""
Jolpica-F1 API Client for Live Race Data Retrieval
Ergast-compatible REST API successor.
"""

import requests
from typing import Dict, Any, Optional

BASE_URL = "http://api.jolpi.ca/ergast/f1"

class JolpicaF1Client:
    def __init__(self, base_url: str = BASE_URL, timeout: int = 10):
        self.base_url = base_url
        self.timeout = timeout
        self.session = requests.Session()

    def get_current_season_races(self) -> Optional[Dict[str, Any]]:
        url = f"{self.base_url}/current.json"
        try:
            res = self.session.get(url, timeout=self.timeout)
            res.raise_for_status()
            return res.json()
        except requests.RequestException as e:
            print(f"API request failed: {e}")
            return None

    def get_race_results(self, season: int, round_num: int) -> Optional[Dict[str, Any]]:
        url = f"{self.base_url}/{season}/{round_num}/results.json"
        try:
            res = self.session.get(url, timeout=self.timeout)
            res.raise_for_status()
            return res.json()
        except requests.RequestException as e:
            print(f"API request failed: {e}")
            return None
