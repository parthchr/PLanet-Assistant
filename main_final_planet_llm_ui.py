# planetAgent_enterprise_final.py
"""
Planet API Assistant (Enterprise Production Edition v3.5 - Final Patch)
=======================================================================
A comprehensive, conversational AI platform for discovering, filtering, 
analyzing, and ordering Planet satellite imagery.

AUTHOR: Senior AI/Python Engineer
DATE: 2026-01-20

SYSTEM ARCHITECTURE:
--------------------
1.  Presentation Layer (Streamlit):
    - Reactive UI with chat bubbles, sidebars, and expanders.
    - Strict separation of system logic from user-facing text.
    - Real-time feedback for long-running processes.

2.  Intelligence Layer (Groq/Llama-3):
    - Context-aware intent extraction.
    - JSON-based structured output generation.
    - Aggressive rate-limit handling (Exponential Backoff).

3.  Geospatial Layer (Geopandas/Shapely):
    - Shapefile ingestion (.zip support).
    - CRS reprojection (EPSG:4326 standardization).
    - **Bounding Box Conversion**: Automatically converts complex shapes to 4-corner bounds.

4.  Data Layer (SQLite):
    - Full schema persistence for metadata.
    - Transactional writes to ensure data integrity.

5.  Integration Layer (Planet API):
    - Data API (Quick Search).
    - Orders API (Clip & Ship).
    - Secure authentication handling.

DEPENDENCIES:
    pip install streamlit requests geopandas shapely geopy python-dotenv

USAGE:
    1. Set PLANET_API_KEY and GROQ_API_KEY in your .env file.
    2. Run: streamlit run planetAgent_enterprise_final.py
"""

import os
import re
import json
import time
import uuid
import logging
import sqlite3
import base64
import zipfile
import tempfile
import folium
from streamlit_folium import st_folium
from shapely.geometry import mapping, shape, Polygon, MultiPolygon
import shutil
import io
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import (
    Optional, 
    Dict, 
    Any, 
    List, 
    Tuple, 
    Union, 
    Callable,
    TypeVar
)
from dataclasses import dataclass, asdict

# Third-party imports
import requests
import streamlit as st
import geopandas as gpd
from shapely.geometry import mapping, shape, Polygon, MultiPolygon, box
from shapely.ops import transform
from geopy.geocoders import Nominatim
from math import cos, radians, sqrt
from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ==================================================================================================
# 1. CONFIGURATION & LOGGING
# ==================================================================================================

# Load environment variables
load_dotenv()

# Configure Application Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("PlanetAssistant")

class AppConfig:
    """
    Central configuration repository for the application.
    """
    # API Credentials
    PLANET_API_KEY: str = os.getenv("PLANET_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    
    # LLM Settings
    LLM_MODEL: str = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    LLM_TEMP: float = float(os.getenv("LLM_TEMP", "0.3")) # Ensure this name matches usage
    LLM_MAX_TOKENS: int = 1000
    
    # Retry Logic Settings (Aggressive for 429)
    MAX_RETRIES: int = 5
    BACKOFF_FACTOR: float = 2.0
    BASE_DELAY: int = 2 
    
    # Database Settings
    DB_PATH: str = "planet_metadata.db"
    
    # Endpoints
    PLANET_DATA_URL: str = "https://api.planet.com/data/v1/quick-search"
    PLANET_ORDERS_URL: str = "https://api.planet.com/compute/ops/orders/v2"
    GROQ_URL: str = "https://api.groq.com/openai/v1/chat/completions"
    OLLAMA_URL: str = "http://localhost:11434/api/generate"
    
    # State Keys
    KEY_HISTORY: str = "chat_history"
    KEY_STATE: str = "assistant_state"
    KEY_FEATURES: str = "features"
    KEY_PREVIEW: str = "active_preview"
    KEY_LAST_UPLOAD: str = "last_uploaded_shp"

    @classmethod
    def validate(cls):
        """Validates critical configuration presence."""
        if not cls.PLANET_API_KEY:
            st.error("CRITICAL: PLANET_API_KEY is missing from environment.")
            st.stop()
        if not cls.GROQ_API_KEY:
            st.warning("WARNING: GROQ_API_KEY is missing. Conversational features will fail.")

# ==================================================================================================
# 2. ROBUST RETRY ENGINE
# ==================================================================================================

def exponential_backoff_retry(max_retries: int = 5, base_delay: int = 2):
    """
    Enterprise-grade decorator to handle API flakiness and Rate Limits (429).
    Implements exponential backoff: 2s, 4s, 8s, 16s, 32s.
    """
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            delay = base_delay
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except requests.exceptions.HTTPError as e:
                    last_exception = e
                    status = e.response.status_code if e.response else 0
                    
                    if status == 429: # Too Many Requests
                        msg = f"⚠️ High traffic (Rate Limit 429). Retrying in {delay}s (Attempt {attempt+1}/{max_retries})..."
                        logger.warning(msg)
                        time.sleep(delay)
                        delay *= 2  # Exponential increase
                        continue
                    elif 500 <= status < 600: # Server Error
                        logger.warning(f"Server error {status}. Retrying...")
                        time.sleep(delay)
                        delay *= 1.5
                        continue
                    else:
                        raise e # Fatal client error (400, 401, 403)
                except Exception as e:
                    # Generic network errors
                    logger.error(f"Network error in {func.__name__}: {str(e)}")
                    time.sleep(delay)
                    delay *= 1.5
                    last_exception = e
                    continue
            
            # If we reach here, we failed
            if last_exception:
                logger.error(f"Max retries exceeded for {func.__name__}: {str(last_exception)}")
                raise last_exception
            return None
        return wrapper
    return decorator

# ==================================================================================================
# 3. DATABASE LAYER
# ==================================================================================================

class DatabaseManager:
    """
    Manages persistent storage of satellite metadata using SQLite.
    Implements the FULL schema as requested.
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_db() # Call init immediately

    def _get_connection(self) -> sqlite3.Connection:
        """Creates a new database connection."""
        return sqlite3.connect(self.db_path)

    def init_db(self):
        """Creates the metadata table if it does not exist."""
        conn = self._get_connection()
        c = conn.cursor()
        
        # Comprehensive Schema Definition
        c.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                id TEXT PRIMARY KEY,
                item_type TEXT,
                acquired TEXT,
                anomalous_pixels REAL,
                clear_confidence_percent REAL,
                clear_percent REAL,
                cloud_cover REAL,
                cloud_percent REAL,
                ground_control BOOLEAN,
                gsd REAL,
                heavy_haze_percent REAL,
                instrument TEXT,
                pixel_resolution REAL,
                provider TEXT,
                published TEXT,
                publishing_stage TEXT,
                quality_category TEXT,
                satellite_azimuth REAL,
                satellite_id TEXT,
                shadow_percent REAL,
                snow_ice_percent REAL,
                strip_id TEXT,
                sun_azimuth REAL,
                sun_elevation REAL,
                updated TEXT,
                view_angle REAL,
                visible_confidence_percent REAL,
                visible_percent REAL,
                geometry TEXT,
                full_metadata TEXT,
                saved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

    def reset_db(self):
        """Destroys and recreates the database."""
        conn = self._get_connection()
        c = conn.cursor()
        c.execute("DROP TABLE IF EXISTS metadata")
        conn.commit()
        conn.close()
        self.init_db()

    def save_features(self, features: List[Dict[str, Any]]):
        """
        Batch inserts a list of GeoJSON features into the database.
        """
        if not features:
            return

        conn = self._get_connection()
        c = conn.cursor()
        
        columns = [
            "id", "item_type", "acquired", "anomalous_pixels", "clear_confidence_percent",
            "clear_percent", "cloud_cover", "cloud_percent", "ground_control", "gsd",
            "heavy_haze_percent", "instrument", "pixel_resolution", "provider",
            "published", "publishing_stage", "quality_category", "satellite_azimuth",
            "satellite_id", "shadow_percent", "snow_ice_percent", "strip_id",
            "sun_azimuth", "sun_elevation", "updated", "view_angle", "visible_confidence_percent",
            "visible_percent", "geometry", "full_metadata"
        ]
        
        placeholders = ",".join(["?"] * len(columns))
        sql = f"INSERT OR REPLACE INTO metadata ({','.join(columns)}) VALUES ({placeholders})"
        
        for item in features:
            try:
                p = item.get("properties", {}) or {}
                geom = item.get("geometry")
                
                values = (
                    item.get("id"),
                    p.get("item_type"),
                    p.get("acquired"),
                    p.get("anomalous_pixels"),
                    p.get("clear_confidence_percent"),
                    p.get("clear_percent"),
                    p.get("cloud_cover"),
                    p.get("cloud_percent"),
                    p.get("ground_control"),
                    p.get("gsd"),
                    p.get("heavy_haze_percent"),
                    p.get("instrument"),
                    p.get("pixel_resolution"),
                    p.get("provider"),
                    p.get("published"),
                    p.get("publishing_stage"),
                    p.get("quality_category"),
                    p.get("satellite_azimuth"),
                    p.get("satellite_id"),
                    p.get("shadow_percent"),
                    p.get("snow_ice_percent"),
                    p.get("strip_id"),
                    p.get("sun_azimuth"),
                    p.get("sun_elevation"),
                    p.get("updated"),
                    p.get("view_angle"),
                    p.get("visible_confidence_percent"),
                    p.get("visible_percent"),
                    json.dumps(geom) if geom else None,
                    json.dumps(item)
                )
                
                c.execute(sql, values)
            except Exception as e:
                logger.error(f"Failed to save item {item.get('id', 'unknown')}: {e}")
                continue
                
        conn.commit()
        conn.close()

# ==================================================================================================
# 4. GEOSPATIAL & ORDERING TOOLS
# ==================================================================================================

def process_uploaded_shapefile(uploaded_file) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Processes an uploaded .zip file containing a Shapefile.
    
    **CRITICAL UPDATE**: This function now automatically calculates the Bounding Box
    (4 corners) of the shapefile to prevent JSON bloat/crashing.
    """
    try:
        with tempfile.TemporaryDirectory() as tmpdirname:
            zip_path = os.path.join(tmpdirname, "upload.zip")
            with open(zip_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Extract
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(tmpdirname)
            
            # Find .shp file
            shp_file = None
            for root, dirs, files in os.walk(tmpdirname):
                for file in files:
                    if file.endswith(".shp"):
                        shp_file = os.path.join(root, file)
                        break
                if shp_file: break
            
            if not shp_file:
                return None, "No .shp file found in the uploaded ZIP."

            # Read and Reproject
            gdf = gpd.read_file(shp_file)
            
            if gdf.crs is not None and gdf.crs.to_string() != "EPSG:4326":
                gdf = gdf.to_crs("EPSG:4326")
            
            if gdf.empty:
                return None, "Shapefile contains no geometries."
            
            # Take the first geometry
            geom = gdf.geometry.iloc[0]
            
            # --- FIX: BOUNDING BOX CONVERSION ---
            # Instead of returning thousands of points, we get the bounds (minx, miny, maxx, maxy)
            # and create a simple rectangular box. This ensures the prompt doesn't crash.
            minx, miny, maxx, maxy = geom.bounds
            
            # Create a clean Polygon from the bounds (5 points: TL, TR, BR, BL, TL)
            bbox_coords = [[
                [minx, miny],
                [maxx, miny],
                [maxx, maxy],
                [minx, maxy],
                [minx, miny]
            ]]
            
            bbox_geojson = {
                "type": "Polygon",
                "coordinates": bbox_coords
            }
            
            return bbox_geojson, None

    except Exception as e:
        return None, str(e)

@exponential_backoff_retry(max_retries=3, base_delay=AppConfig.BASE_DELAY)
def place_planet_order(scene_ids: List[str], aoi_geometry: Dict, order_name: str = "Composite Order") -> Dict:
    """
    Submits a Clip & Ship (Composite) order to Planet Orders v2 API.
    """
    if not AppConfig.PLANET_API_KEY:
        return {"success": False, "error": "PLANET_API_KEY not configured"}

    # Define Tools: Clip + Composite
    tools = [
        {
            "clip": {
                "aoi": aoi_geometry
            }
        },
        {
            "composite": {}
        }
    ]

    payload = {
        "name": order_name,
        "source_type": "scenes",
        "products": [
            {
                "item_ids": scene_ids,
                "item_type": "PSScene",
                "product_bundle": "analytic_udm2" 
            }
        ],
        "tools": tools
    }

    try:
        response = requests.post(
            AppConfig.PLANET_ORDERS_URL,
            json=payload,
            auth=HTTPBasicAuth(AppConfig.PLANET_API_KEY, ""),
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code in [200, 202]:
            return {"success": True, "data": response.json()}
        else:
            return {"success": False, "error": f"Planet API Error ({response.status_code}): {response.text}"}
            
    except Exception as e:
        return {"success": False, "error": str(e)}

# ==================================================================================================
# 5. INTELLIGENCE LAYER (LLM & AGENT)
# ==================================================================================================

class LLMExtractor:
    """
    Interacts with the Groq/Llama API to extract structured parameters.
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = AppConfig.LLM_MODEL
        self.temp = AppConfig.LLM_TEMP # FIXED: Matches AppConfig attribute name

    @exponential_backoff_retry(max_retries=AppConfig.MAX_RETRIES, base_delay=AppConfig.BASE_DELAY)
    def extract_and_reply(self, user_message: str, recent_history: List[Dict], assistant_state: Dict) -> Dict:
        """
        Sends context to LLM and retrieves JSON decision block.
        """
        if not self.api_key:
            raise RuntimeError("LLM API key missing")
        
        system_prompt = (
            "You are a warm, human receptionist-style assistant for Planet satellite imagery. "
            "Your goal is to collect these 4 filters: start_date, end_date, cloud_cover, geometry. "
            "\n\nRULES:"
            "\n1. Check 'assistant_state'. If 'geometry' is present (e.g. from Shapefile), DO NOT ask for location."
            "\n2. If data is missing, ask politely for it."
            "\n3. If user says 'assume' or 'I don't have coords', set decision='defaulted'."
            "\n4. If the user gives a location name, extract it into 'place'."
            "\n\nOUTPUT FORMAT (STRICT JSON):"
            "\n{"
            "\n  \"start_date\": \"YYYY-MM-DD\" or null,"
            "\n  \"end_date\": \"YYYY-MM-DD\" or null,"
            "\n  \"cloud_cover\": \"< 0.X\" or null,"
            "\n  \"geometry\": object or null,"
            "\n  \"place\": string or null,"
            "\n  \"decision\": \"complete\" | \"ask\" | \"defaulted\","
            "\n  \"reply\": \"Your friendly message to the user here. DO NOT include JSON here.\""
            "\n}"
        )

        # Build Context
        messages = [{"role": "system", "content": system_prompt}]
        for h in (recent_history or [])[-6:]:
            messages.append(h)
            
        # Add Current State
        state_context = f"assistant_state = {json.dumps(assistant_state)}\n\nuser_message = {json.dumps(user_message)}"
        messages.append({"role": "user", "content": state_context})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temp,
            "max_tokens": AppConfig.LLM_MAX_TOKENS,
            "response_format": {"type": "json_object"}
        }

        url = AppConfig.GROQ_URL
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        
        response = requests.post(url, headers=headers, json=payload, timeout=40)
        response.raise_for_status()
            
        response_data = response.json()
        content = response_data["choices"][0]["message"]["content"]

        # Parse & Clean
        parsed = None
        try:
            parsed, substr = extract_json_from_text(content)
            if isinstance(parsed, list):
                parsed = parsed[0]
        except Exception:
            parsed = {"decision": "ask", "reply": content}

        # --- CRITICAL UI FIX ---
        assistant_text = parsed.get("reply", "")
        
        # Regex strip of top-level JSON objects if LLM leaked them
        if "{" in assistant_text and "}" in assistant_text:
             assistant_text = re.sub(r'\{.*?\}', '', assistant_text, flags=re.DOTALL)
        
        assistant_text = assistant_text.strip()
        
        if not assistant_text or assistant_text.lower() == "null":
            assistant_text = "I've noted that. Is there anything else you'd like to add?"

        parsed["reply"] = assistant_text
        
        # Sanitize Keys
        for k in ["start_date", "end_date", "cloud_cover", "geometry", "place"]:
            parsed.setdefault(k, None)
            
        if parsed.get("decision") not in ["complete", "ask", "defaulted"]:
            parsed["decision"] = "ask"

        return {"assistant_text": assistant_text, "parsed": parsed}

class PlanetAIAgent:
    """Orchestrates conversation, state, and API calls."""
    def __init__(self, llm_api_key: str):
        self.llm = LLMExtractor(llm_api_key)
        self.geolocator = Nominatim(user_agent="planet-assistant-v2")
        self.db_manager = DatabaseManager(AppConfig.DB_PATH)

    def geocode_place(self, place_name: str) -> Optional[Dict]:
        try:
            loc = self.geolocator.geocode(place_name, addressdetails=True)
            if not loc: return None
            
            raw = getattr(loc, "raw", {})
            bbox = None
            if "boundingbox" in raw:
                try:
                    s, n, w, e = float(raw["boundingbox"][0]), float(raw["boundingbox"][1]), float(raw["boundingbox"][2]), float(raw["boundingbox"][3])
                    bbox = (s, w, n, e)
                except ValueError: pass
                    
            return {"lat": loc.latitude, "lon": loc.longitude, "bbox": bbox, "display_name": raw.get("display_name")}
        except Exception as e:
            print(f"Geocoding error: {e}")
            return None

    def search_planet_metadata(self, filters: Dict) -> List[Dict]:
        if not AppConfig.PLANET_API_KEY:
            raise RuntimeError("PLANET_API_KEY not configured")
            
        if isinstance(filters.get("geometry"), str):
            g = parse_geometry_input(filters["geometry"])
            if g: filters["geometry"] = g
                
        body = build_planet_api_body(filters)
        
        if not body["filter"]["config"]:
            raise ValueError("No valid filters provided for search.")
        
        auth = HTTPBasicAuth(AppConfig.PLANET_API_KEY, "")
        headers = {"Content-Type": "application/json"}
        
        response = requests.post(AppConfig.PLANET_DATA_URL, auth=auth, headers=headers, json=body, timeout=90)
        response.raise_for_status()
        
        data = response.json()
        features = data.get("features", [])
        
        self.db_manager.save_features(features)
        return features

    def handle_user_prompt(self, user_prompt: str):
        # Session Init
        if "assistant_state" not in st.session_state:
            st.session_state.assistant_state = {"start_date": None, "end_date": None, "cloud_cover": None, "geometry": None, "place": None}
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        # Cleanup old results
        if "features" in st.session_state: del st.session_state.features
        if "active_preview" in st.session_state: del st.session_state.active_preview

        # 1. Update History
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})

        # 2. Heuristic: Direct Coordinate Detection
        geom_direct = parse_geometry_input(user_prompt)
        if geom_direct:
            st.session_state.assistant_state["geometry"] = geom_direct 

        # ==================================================================================
        # 🚨 PRE-LLM BYPASS RULE (FORCE READY)
        # ==================================================================================
        state = st.session_state.assistant_state

        # Check if we have the Big 3 (Dates + Geom) BEFORE calling LLM
        if state.get("start_date") and state.get("end_date") and state.get("geometry"):
            assistant_text = "I have the location and dates. Searching Planet's archive now."
            
            # Ensure history is updated manually since we skip LLM
            st.session_state.chat_history.append({"role": "assistant", "content": assistant_text})

            filters = {
                "start_date": state.get("start_date"),
                "end_date": state.get("end_date"),
                "cloud_cover": state.get("cloud_cover"),
                "geometry": state.get("geometry")
            }

            return {
                "status": "ready",
                "assistant_text": assistant_text,
                "filters": filters
            }

        # 3. LLM Extraction (Only if missing data)
        try:
            out = self.llm.extract_and_reply(user_prompt, st.session_state.chat_history, st.session_state.assistant_state)
        except Exception as e:
            error_msg = f"I'm having trouble connecting to my brain right now (Error: {str(e)}). Please try asking again in a moment."
            st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
            return {"status": "error", "assistant_text": error_msg}

        parsed = out["parsed"]
        assistant_text = out["assistant_text"]
        
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_text})

        # 4. Update State
        state = st.session_state.assistant_state
        for k in ["start_date", "end_date", "cloud_cover", "geometry", "place"]:
            if parsed.get(k): state[k] = parsed.get(k)

        # 5. Geocoding / Defaulting Logic
        if parsed.get("decision") == "defaulted" and not state.get("geometry"):
            place = parsed.get("place") or state.get("place")
            if place:
                geo = self.geocode_place(place)
                if geo and geo.get("lat"):
                    half_km = sqrt(30) / 2
                    state["geometry"] = create_small_bbox_polygon_from_point(geo["lat"], geo["lon"], half_km)

        # 6. Check Completion
        is_ready = bool(state.get("start_date") and state.get("end_date") and state.get("geometry"))
        
        if parsed.get("decision") == "complete" or is_ready:
            filters = {
                "start_date": state.get("start_date"),
                "end_date": state.get("end_date"),
                "cloud_cover": state.get("cloud_cover"),
                "geometry": state.get("geometry")
            }
            return {"status": "ready", "assistant_text": assistant_text, "filters": filters}

        return {"status": "need_clarify", "assistant_text": assistant_text, "missing": parsed.get("clarify")}

def build_planet_api_body(filters: Dict) -> Dict:
    body = {"item_types": ["PSScene"], "filter": {"type": "AndFilter", "config": []}}
    start = _normalize_date_iso(filters.get("start_date"), "start")
    end = _normalize_date_iso(filters.get("end_date"), "end")
    if start and end and start > end: start, end = end, start
    
    date_config = {}
    if start: date_config["gte"] = start
    if end: date_config["lte"] = end
    
    if date_config:
        body["filter"]["config"].append({"type": "DateRangeFilter", "field_name": "acquired", "config": date_config})

    cloud = _normalize_cloud_cover(filters.get("cloud_cover"))
    if cloud is not None:
        body["filter"]["config"].append({"type": "RangeFilter", "field_name": "cloud_cover", "config": {"lte": cloud}})
    
    geom = filters.get("geometry")
    if geom:
        body["filter"]["config"].append({"type": "GeometryFilter", "field_name": "geometry", "config": geom})
    
    return body

# --- PREVIEW & VLM HELPERS ---
def fetch_thumbnail(thumbnail_url: str, api_key: str) -> Optional[bytes]:
    try:
        auth = HTTPBasicAuth(api_key, "")
        response = requests.get(thumbnail_url, auth=auth, timeout=30)
        response.raise_for_status()
        return response.content
    except Exception as e:
        print(f"Thumbnail error: {e}")
        return None

def get_vlm_summary(image_bytes: bytes) -> str:
    try:
        encoded_image = base64.b64encode(image_bytes).decode('utf-8')
        payload = {"model": "llava", "prompt": "Describe this satellite image in a single, concise paragraph.", "images": [encoded_image], "stream": False}
        response = requests.post(AppConfig.OLLAMA_URL, json=payload, timeout=60)
        response.raise_for_status()
        return response.json().get("response", "No summary generated.").strip()
    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to local Ollama. Is it running?"
    except Exception as e:
        return f"VLM Error: {str(e)}"

# --- RESTORED UTILITIES ---
def _normalize_date_iso(date_str: str, which: str = "start") -> Optional[str]:
    if not date_str: return None
    s = str(date_str).strip()
    if "T" in s: return s if s.endswith("Z") else s + "Z"
    m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", s)
    if m:
        year, month, day = m.group(1), m.group(2), m.group(3)
        suffix = "T00:00:00.000Z" if which == "start" else "T23:59:59.999Z"
        return f"{year}-{int(month):02d}-{int(day):02d}{suffix}"
    return s

def _normalize_cloud_cover(val: Any) -> Optional[float]:
    if val is None: return None
    try:
        v = float(val)
    except:
        m = re.search(r"(\d+(\.\d+)?)", str(val))
        if not m: return None
        v = float(m.group(1))
    if v > 1.0: v = v / 100.0
    return max(0.0, min(1.0, v))

def parse_geometry_input(value: Any) -> Optional[Dict]:
    if not value: return None
    if isinstance(value, dict): return value
    s = str(value).strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and obj.get("type") and obj.get("coordinates"): return obj
    except: pass
    m = re.search(r"\[?\s*(-?\d+(?:\.\d+)?)\s*[,\s]\s*(-?\d+(?:\.\d+)?)\s*[,\s]\s*(-?\d+(?:\.\d+)?)\s*[,\s]\s*(-?\d+(?:\.\d+)?)\s*\]?", s)
    if m:
        coords = [float(x) for x in m.groups()]
        min_lon, min_lat, max_lon, max_lat = coords
        if min_lon > max_lon: min_lon, max_lon = max_lon, min_lon
        if min_lat > max_lat: min_lat, max_lat = max_lat, min_lat
        return {"type": "Polygon", "coordinates": [[[min_lon, min_lat], [max_lon, min_lat], [max_lon, max_lat], [min_lon, max_lat], [min_lon, min_lat]]]}
    return None

def area_km2_from_bbox(min_lat, min_lon, max_lat, max_lon):
    center_lat = (min_lat + max_lat) / 2.0
    height_deg = max_lat - min_lat
    width_deg = max_lon - min_lon
    height_km = abs(height_deg) * 111.0
    width_km = abs(width_deg) * 111.0 * abs(cos(radians(center_lat)))
    return abs(width_km * height_km)

def create_small_bbox_polygon_from_point(lat: float, lon: float, half_km: float = 2.74) -> Dict:
    deg_lat = half_km / 111.0
    deg_lon = half_km / (111.0 * abs(cos(radians(lat)) or 1.0))
    min_lon = lon - deg_lon
    max_lon = lon + deg_lon
    min_lat = lat - deg_lat
    max_lat = lat + deg_lat
    coords = [[[min_lon, min_lat], [max_lon, min_lat], [max_lon, max_lat], [min_lon, max_lat], [min_lon, min_lat]]]
    return {"type": "Polygon", "coordinates": coords}

def find_first_json_substring(text: str) -> Optional[str]:
    if not text: return None
    length = len(text)
    for i, ch in enumerate(text):
        if ch not in ('{', '['): continue
        start = i; stack = [ch]; in_str = False; esc = False
        for j in range(i + 1, length):
            c = text[j]
            if esc: esc = False; continue
            if c == '\\': esc = True; continue
            if c == '"' and not esc: in_str = not in_str; continue
            if in_str: continue
            if c == '{': stack.append('{')
            elif c == '[': stack.append('[')
            elif c == '}' and stack and stack[-1] == '{':
                stack.pop()
                if not stack: return text[start:j + 1]
            elif c == ']' and stack and stack[-1] == '[':
                stack.pop()
                if not stack: return text[start:j + 1]
    return None

def extract_json_from_text(text: str) -> Tuple[Optional[Union[Dict, List]], str]:
    if text is None: raise ValueError("Empty text")
    try:
        return json.loads(text), text
    except: pass
    substr = find_first_json_substring(text)
    if not substr: raise ValueError("No JSON found")
    return json.loads(substr), substr

# ==================================================================================================
# 7. MAIN UI APPLICATION
# ==================================================================================================

def main():
    AppConfig.validate()
    
    # Initialize DB
    db = DatabaseManager(AppConfig.DB_PATH)
    # db.init_db() is called inside __init__, so we don't need to call it again
    
    # Initialize Agent
    agent = PlanetAIAgent(AppConfig.GROQ_API_KEY)
    st.title("🌍 Planet API Assistant")

    # Initialize Session State
    if "assistant_state" not in st.session_state:
        st.session_state.assistant_state = {
            "start_date": None, "end_date": None, "cloud_cover": None, "geometry": None, "place": None
        }
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # ----------------------------------------------------------------------------------------------
    # SIDEBAR CONTROLS
    # ----------------------------------------------------------------------------------------------
    st.sidebar.title("🛠️ Tools")
    
    # --- Feature 1: Shapefile Upload ---
    st.sidebar.subheader("1. Area Selection")
    st.sidebar.markdown("Upload a `.zip` containing your Shapefile (.shp, .shx, .dbf) to automatically set the geometry.")
    
    uploaded_file = st.sidebar.file_uploader("Upload Shapefile", type="zip", key="shp_upload")
    
    if uploaded_file is not None:
        # Prevent re-processing on every rerun if file hasn't changed
        if st.session_state.get("last_uploaded_shp") != uploaded_file.name:
            with st.sidebar.status("Processing Shapefile..."):
                geom_data, error_msg = process_uploaded_shapefile(uploaded_file)
                
                if geom_data:
                    # 1. Update State
                    st.session_state.assistant_state["geometry"] = geom_data
                    st.session_state["last_uploaded_shp"] = uploaded_file.name
                    
                    # 2. Context Injection (Hidden from user, seen by LLM)
                    st.session_state.chat_history.append({
                        "role": "user", 
                        "content": "System Update: I have uploaded a Shapefile. The geometry is now set."
                    })
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": "I've received your Shapefile and set the Area of Interest. Please provide dates or cloud cover preferences if you haven't already."
                    })
                    st.success("Shapefile loaded successfully!")
                    st.rerun()
                else:
                    st.error(f"Shapefile Error: {error_msg}")

    st.sidebar.markdown("---")
    
    # Reset Button
    if st.sidebar.button("🗑️ Start New Chat"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        db.reset_db()
        st.rerun()
        
    st.sidebar.caption(f"Model: {AppConfig.LLM_MODEL}")

    # ----------------------------------------------------------------------------------------------
    # CHAT INTERFACE
    # ----------------------------------------------------------------------------------------------
    
    # Display Chat History
    # We filter out "System Update" messages to keep the UI clean as requested
    for msg in st.session_state.chat_history:
        if not msg["content"].startswith("System Update"):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # User Input
    if user_text := st.chat_input("Ask about satellite imagery (e.g., 'Show me Tokyo in June 2024')..."):
        
        # Process the input
        result = agent.handle_user_prompt(user_text)
        
        # Force a UI update to show the new messages immediately
        # Note: We display the last 2 messages (User + Assistant)
        if len(st.session_state.chat_history) >= 2:
            with st.chat_message("user"):
                st.markdown(st.session_state.chat_history[-2]['content'])
            with st.chat_message("assistant"):
                st.markdown(st.session_state.chat_history[-1]['content'])
        
        # ------------------------------------------------------------------------------------------
        # ACTION: SEARCH TRIGGER
        # ------------------------------------------------------------------------------------------
        if result.get("status") == "ready":
            filters = result["filters"]
            
            # Show the "Filters Set" Green Box (Success UI)
            with st.chat_message("assistant"):
                st.success("✅ Filters set! I'm searching Planet's archive now...")
                with st.expander("View Filter Details", expanded=True):
                    st.code(json.dumps(filters, indent=2), language='json')
            
            # Execute Search
            try:
                with st.spinner("Querying Planet API..."):
                    features = agent.search_planet_metadata(filters)
                    st.session_state.features = features
                    
                    if features:
                        st.success(f"Found {len(features)} images matching your criteria.")
                    else:
                        st.warning("No images found for these filters. Try widening your date range or cloud cover.")
                        
            except Exception as e:
                st.error(f"Search failed: {str(e)}")
        
        elif result.get("status") == "error":
            st.error(result.get("assistant_text"))

    # ----------------------------------------------------------------------------------------------
    # RESULTS & ADVANCED FEATURES
    # ----------------------------------------------------------------------------------------------
    if "features" in st.session_state and st.session_state.features:
        features = st.session_state.features
        
        # --- Feature 2: Clip & Ship Ordering ---
        st.markdown("---")
        st.subheader("📦 Order & Process")
        st.caption("Select scenes below to create a clipped composite image based on your Area of Interest.")
        
        with st.container(border=True):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Limit options to 50 to prevent UI lag
                available_ids = [f["id"] for f in features[:50]]
                selected_ids = st.multiselect("Select Scenes to Composite:", options=available_ids)
            
            with col2:
                st.write("") # Spacing
                st.write("") 
                aoi = st.session_state.assistant_state.get("geometry")
                
                # Validation for Ordering
                is_ready_to_order = bool(selected_ids and aoi)
                
                if st.button("🚀 Place Clip Order", disabled=not is_ready_to_order, use_container_width=True):
                    if not aoi:
                        st.error("No AOI found. Upload a Shapefile or define a location.")
                    else:
                        with st.spinner("Submitting Order to Planet..."):
                            order_name = f"Composite_{len(selected_ids)}_scenes_{int(time.time())}"
                            order_result = place_planet_order(selected_ids, aoi, order_name)
                            
                            if order_result["success"]:
                                st.balloons()
                                st.success("Order Placed Successfully!")
                                st.json(order_result["data"])
                            else:
                                st.error(f"Order Failed: {order_result['error']}")

        # --- Results Table ---
        st.markdown("### Search Results")
        
        preview_list = []
        for f in features[:50]: # Show top 50
            props = f.get("properties", {})
            links = f.get("_links", {})
            preview_list.append({
                "id": f.get("id"),
                "acquired": props.get("acquired"),
                "cloud": props.get("cloud_cover"),
                "satellite": props.get("satellite_id") or props.get("item_type"),
                "thumbnail": links.get("thumbnail")
            })
        
        # Table Header
        h1, h2, h3, h4, h5 = st.columns([3, 2, 2, 2, 1])
        h1.markdown("**Scene ID**")
        h2.markdown("**Date**")
        h3.markdown("**Cloud Cover**")
        h4.markdown("**Satellite**")
        h5.markdown("**View**")
        
        # Table Rows
        for item in preview_list:
            c1, c2, c3, c4, c5 = st.columns([3, 2, 2, 2, 1])
            c1.write(item["id"])
            c2.write(item["acquired"])
            c3.write(f"{item['cloud']:.2%}" if item['cloud'] is not None else "N/A")
            c4.write(item["satellite"])
            
            # Preview Button
            if c5.button("👁️", key=f"btn_{item['id']}"):
                if item["thumbnail"]:
                    with st.spinner("Fetching preview and analyzing with AI..."):
                        img_bytes = fetch_thumbnail(item["thumbnail"], AppConfig.PLANET_API_KEY)
                        
                        if img_bytes:
                            summary = get_vlm_summary(img_bytes)
                            st.session_state.active_preview = {
                                "id": item["id"],
                                "img": img_bytes,
                                "sum": summary
                            }
                            st.rerun()
                        else:
                            st.error("Thumbnail fetch failed.")
                else:
                    st.warning("No thumbnail available.")

        # --- Active Preview Expander ---
        if "active_preview" in st.session_state:
            p = st.session_state.active_preview
            st.markdown("---")
            with st.expander(f"📷 Analysis for {p['id']}", expanded=True):
                col_img, col_txt = st.columns([1, 1])
                with col_img:
                    st.image(p["img"], caption=f"Scene {p['id']}", use_column_width=True)
                with col_txt:
                    st.markdown("### 🤖 AI Summary")
                    st.info(p["sum"]) 

if __name__ == "__main__":
    main()