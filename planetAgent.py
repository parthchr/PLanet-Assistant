# planetAgent_enterprise_final.py
"""
Planet API Assistant (Enterprise Edition v4.0 - Final Production)
=================================================================
A comprehensive AI platform for satellite imagery discovery, analysis, and ordering.

FEATURES:
1.  **Conversational Interface**: Groq (Llama 3) for intent understanding.
2.  **Shapefile Handler**: Auto-converts uploaded .zip shapefiles to Bounding Boxes to prevent API crashes.
3.  **Interactive Map**: Visualizes AOI (Blue) vs. Scene Footprints (Orange) using Folium.
4.  **Metadata Analyst**: Generates AI summaries based on DB metadata (Sun Azimuth, View Angle, etc.) instead of raw pixels.
5.  **Clip & Ship**: Orders composite assets via Planet Orders v2.
6.  **Robust Core**: Exponential backoff retries, transactional DB, and strict UI hygiene.

DEPENDENCIES:
    pip install streamlit requests geopandas shapely geopy python-dotenv folium streamlit-folium

USAGE:
    streamlit run planetAgent_enterprise_final.py
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
    Callable
)

# Third-party imports
import requests
import streamlit as st
import geopandas as gpd
import folium
from streamlit_folium import st_folium
from shapely.geometry import mapping, shape, Polygon
from geopy.geocoders import Nominatim
from math import cos, radians, sqrt
from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth

# ==================================================================================================
# 1. CONFIGURATION & LOGGING
# ==================================================================================================

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("PlanetAssistant")

class AppConfig:
    """Central configuration repository."""
    # API Credentials
    PLANET_API_KEY: str = os.getenv("PLANET_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    
    # AI Settings
    LLM_MODEL: str = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    LLM_TEMP: float = float(os.getenv("LLM_TEMP", "0.3"))
    LLM_MAX_TOKENS: int = 1000
    
    # Retry Settings
    MAX_RETRIES: int = 5
    BACKOFF_FACTOR: float = 2.0
    BASE_DELAY: int = 2  # Fixed: Defined to prevent AttributeError
    
    # Paths & URLs
    DB_PATH: str = "planet_metadata.db"
    PLANET_DATA_URL: str = "https://api.planet.com/data/v1/quick-search"
    PLANET_ORDERS_URL: str = "https://api.planet.com/compute/ops/orders/v2"
    GROQ_URL: str = "https://api.groq.com/openai/v1/chat/completions"
    
    # State Management Keys
    KEY_HISTORY: str = "chat_history"
    KEY_STATE: str = "assistant_state"
    KEY_FEATURES: str = "features"
    KEY_PREVIEW: str = "active_preview"
    KEY_LAST_UPLOAD: str = "last_uploaded_shp"

    @classmethod
    def validate(cls):
        if not cls.PLANET_API_KEY:
            st.error("CRITICAL: PLANET_API_KEY missing.")
            st.stop()
        if not cls.GROQ_API_KEY:
            st.warning("WARNING: GROQ_API_KEY missing.")

# ==================================================================================================
# 2. ROBUST RETRY ENGINE
# ==================================================================================================

def exponential_backoff_retry(max_retries: int = 5, base_delay: int = 2):
    """
    Decorator to handle API Rate Limits (429) with exponential backoff.
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
                    
                    if status == 429:
                        logger.warning(f"Rate limit 429. Retrying in {delay}s...")
                        time.sleep(delay)
                        delay *= 2
                        continue
                    elif 500 <= status < 600:
                        time.sleep(delay)
                        delay *= 1.5
                        continue
                    else:
                        raise e
                except Exception as e:
                    logger.error(f"Error in {func.__name__}: {e}")
                    time.sleep(delay)
                    delay *= 1.5
                    last_exception = e
                    continue
            
            if last_exception:
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
    """
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_schema() # Auto-initialize on creation

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_schema(self):
        """Initializes the full 30-column database schema."""
        conn = self._get_connection()
        c = conn.cursor()
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

    def reset_database(self):
        """Destroys and recreates the database."""
        conn = self._get_connection()
        c = conn.cursor()
        c.execute("DROP TABLE IF EXISTS metadata")
        conn.commit()
        conn.close()
        self._init_schema()

    def get_metadata_by_id(self, scene_id: str) -> Optional[Dict]:
        """Retrieves full metadata for a specific scene ID."""
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM metadata WHERE id = ?", (scene_id,))
        row = c.fetchone()
        conn.close()
        return dict(row) if row else None

    def save_features(self, features: List[Dict[str, Any]]):
        """Batch inserts GeoJSON features."""
        if not features: return
        conn = self._get_connection()
        c = conn.cursor()
        
        cols = [
            "id", "item_type", "acquired", "anomalous_pixels", "clear_confidence_percent",
            "clear_percent", "cloud_cover", "cloud_percent", "ground_control", "gsd",
            "heavy_haze_percent", "instrument", "pixel_resolution", "provider",
            "published", "publishing_stage", "quality_category", "satellite_azimuth",
            "satellite_id", "shadow_percent", "snow_ice_percent", "strip_id",
            "sun_azimuth", "sun_elevation", "updated", "view_angle", "visible_confidence_percent",
            "visible_percent", "geometry", "full_metadata"
        ]
        placeholders = ",".join(["?"] * len(cols))
        sql = f"INSERT OR REPLACE INTO metadata ({','.join(cols)}) VALUES ({placeholders})"
        
        for item in features:
            try:
                p = item.get("properties", {}) or {}
                geom = item.get("geometry")
                
                row = (
                    item.get("id"), p.get("item_type"), p.get("acquired"), p.get("anomalous_pixels"),
                    p.get("clear_confidence_percent"), p.get("clear_percent"), p.get("cloud_cover"),
                    p.get("cloud_percent"), p.get("ground_control"), p.get("gsd"), p.get("heavy_haze_percent"),
                    p.get("instrument"), p.get("pixel_resolution"), p.get("provider"), p.get("published"),
                    p.get("publishing_stage"), p.get("quality_category"), p.get("satellite_azimuth"),
                    p.get("satellite_id"), p.get("shadow_percent"), p.get("snow_ice_percent"), p.get("strip_id"),
                    p.get("sun_azimuth"), p.get("sun_elevation"), p.get("updated"), p.get("view_angle"),
                    p.get("visible_confidence_percent"), p.get("visible_percent"),
                    json.dumps(geom) if geom else None, json.dumps(item)
                )
                c.execute(sql, row)
            except Exception:
                continue
        conn.commit()
        conn.close()

# ==================================================================================================
# 4. GEOSPATIAL PROCESSING & MAPPING
# ==================================================================================================

class GeoProcessor:
    """Handles geometry logic, shapefiles, and Map Rendering."""

    @staticmethod
    def process_uploaded_shapefile(uploaded_file) -> Tuple[Optional[Dict], Optional[str]]:
        """
        Processes uploaded .zip shapefile.
        **CRITICAL:** Converts complex shapes to Bounding Box to prevent JSON/LLM crashes.
        """
        try:
            with tempfile.TemporaryDirectory() as tmpdirname:
                zip_path = os.path.join(tmpdirname, "upload.zip")
                with open(zip_path, "wb") as f: f.write(uploaded_file.getbuffer())
                with zipfile.ZipFile(zip_path, 'r') as zip_ref: zip_ref.extractall(tmpdirname)
                
                shp_file = None
                for root, dirs, files in os.walk(tmpdirname):
                    for file in files:
                        if file.endswith(".shp"):
                            shp_file = os.path.join(root, file); break
                    if shp_file: break
                
                if not shp_file: return None, "No .shp file found."

                gdf = gpd.read_file(shp_file)
                if gdf.crs is not None and gdf.crs.to_string() != "EPSG:4326": gdf = gdf.to_crs("EPSG:4326")
                if gdf.empty: return None, "Shapefile is empty."
                
                geom = gdf.geometry.iloc[0]
                
                # BBOX CONVERSION (Crucial for stability)
                minx, miny, maxx, maxy = geom.bounds
                bbox_coords = [[[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy], [minx, miny]]]
                return {"type": "Polygon", "coordinates": bbox_coords}, None

        except Exception as e:
            return None, str(e)

    @staticmethod
    def render_search_map(aoi_geom: Dict, features: List[Dict]):
        """
        Creates a Folium map visualizing the AOI (Blue) and Scene Footprints (Orange).
        """
        try:
            # 1. Center Map
            s = shape(aoi_geom)
            centroid = s.centroid
            m = folium.Map(location=[centroid.y, centroid.x], zoom_start=10, tiles="OpenStreetMap")

            # 2. Add AOI (Blue)
            folium.GeoJson(
                aoi_geom,
                name="Your Area",
                style_function=lambda x: {'color': 'blue', 'fillColor': 'blue', 'fillOpacity': 0.1, 'weight': 2}
            ).add_to(m)

            # 3. Add Scenes (Orange) - Limit to top 20
            for f in features[:20]:
                fid = f.get('id')
                date = f.get('properties', {}).get('acquired')
                folium.GeoJson(
                    f['geometry'],
                    name=fid,
                    tooltip=f"ID: {fid}\nDate: {date}",
                    style_function=lambda x: {'color': 'orange', 'fillColor': 'orange', 'fillOpacity': 0.05, 'weight': 1}
                ).add_to(m)
            
            return m
        except Exception as e:
            st.error(f"Map Error: {e}")
            return None

    @staticmethod
    def parse_geometry_input(value: Any) -> Optional[Dict]:
        """Robust parser for geometry inputs."""
        if not value: return None
        if isinstance(value, dict): return value
        s = str(value).strip()
        try: return json.loads(s)
        except: pass
        # Regex for raw coordinate lists
        m = re.search(r"\[?\s*(-?\d+(?:\.\d+)?)\s*[,\s]\s*(-?\d+(?:\.\d+)?)\s*[,\s]\s*(-?\d+(?:\.\d+)?)\s*[,\s]\s*(-?\d+(?:\.\d+)?)\s*\]?", s)
        if m:
            coords = [float(x) for x in m.groups()]
            min_lon, min_lat, max_lon, max_lat = coords
            # Enforce Min/Max order
            if min_lon > max_lon: min_lon, max_lon = max_lon, min_lon
            if min_lat > max_lat: min_lat, max_lat = max_lat, min_lat
            return {"type": "Polygon", "coordinates": [[[min_lon, min_lat], [max_lon, min_lat], [max_lon, max_lat], [min_lon, max_lat], [min_lon, min_lat]]]}
        return None

    @staticmethod
    def create_small_bbox_polygon_from_point(lat: float, lon: float, half_km: float = 2.74) -> Dict:
        deg_lat = half_km / 111.0
        deg_lon = half_km / (111.0 * abs(cos(radians(lat)) or 1.0))
        min_lon, max_lon = lon - deg_lon, lon + deg_lon
        min_lat, max_lat = lat - deg_lat, lat + deg_lat
        return {"type": "Polygon", "coordinates": [[[min_lon, min_lat], [max_lon, min_lat], [max_lon, max_lat], [min_lon, max_lat], [min_lon, min_lat]]]}

# ==================================================================================================
# 5. INTELLIGENCE LAYER (LLM & AGENT)
# ==================================================================================================

class LLMEngine:
    """Handles interactions with Groq for conversation and metadata analysis."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = AppConfig.LLM_MODEL
        self.temp = AppConfig.LLM_TEMP

    @exponential_backoff_retry(max_retries=AppConfig.MAX_RETRIES, base_delay=AppConfig.BASE_DELAY)
    def extract_filters(self, user_message: str, recent_history: List[Dict], assistant_state: Dict) -> Dict:
        """Standard conversational extraction logic."""
        if not self.api_key: raise RuntimeError("LLM API key missing")
        
        system_prompt = (
            "You are a warm, human receptionist-style assistant for Planet satellite imagery. "
            "Your goal is to collect these 4 filters: start_date, end_date, cloud_cover, geometry. "
            "\nRULES:"
            "\n1. If 'geometry' is in 'assistant_state' (e.g. from Shapefile), DO NOT ask for location."
            "\n2. If user implies 'no coordinates', set decision='defaulted'."
            "\n\nOUTPUT FORMAT (STRICT JSON):"
            "\n{"
            "\n  \"start_date\": \"YYYY-MM-DD\" or null,"
            "\n  \"end_date\": \"YYYY-MM-DD\" or null,"
            "\n  \"cloud_cover\": \"< 0.X\" or null,"
            "\n  \"geometry\": object or null,"
            "\n  \"place\": string or null,"
            "\n  \"decision\": \"complete\" | \"ask\" | \"defaulted\","
            "\n  \"reply\": \"Friendly message to user. NO JSON.\""
            "\n}"
        )

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(recent_history[-6:])
        messages.append({
            "role": "user",
            "content": f"assistant_state = {json.dumps(assistant_state)}\nuser_message = {user_message}"
        })

        payload = {
            "model": self.model, "messages": messages, "temperature": self.temp,
            "max_tokens": 800, "response_format": {"type": "json_object"}
        }
        
        url = AppConfig.GROQ_URL
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        response = requests.post(url, headers=headers, json=payload, timeout=40)
        response.raise_for_status()
        
        content = response.json()["choices"][0]["message"]["content"]
        
        try:
            parsed, _ = self._extract_json_from_text(content)
        except:
            parsed = {"decision": "ask", "reply": content}

        # UI Scrubbing
        reply = parsed.get("reply", "")
        if "{" in reply: reply = re.sub(r'\{.*?\}', '', reply, flags=re.DOTALL)
        parsed["reply"] = reply.strip() or "I've noted that. What else?"
        
        # Sanitize
        for k in ["start_date", "end_date", "cloud_cover", "geometry", "place"]: parsed.setdefault(k, None)
        if parsed.get("decision") not in ["complete", "ask", "defaulted"]: parsed["decision"] = "ask"
        return {"assistant_text": parsed["reply"], "parsed": parsed}

    @exponential_backoff_retry(max_retries=3, base_delay=2)
    def analyze_metadata(self, metadata: Dict) -> str:
        """
        NEW FEATURE: Generates a professional summary based on DB metadata columns.
        """
        system_prompt = (
            "You are a specialized Satellite Imagery Analyst. "
            "You will be given raw metadata for a satellite scene. "
            "Write a concise, professional summary (bullet points) highlighting key quality metrics. "
            "Focus on: Acquired Date, Cloud Cover, Sun Elevation, Off-Nadir Angle, and Ground Control. "
            "Do NOT hallucinate features not present in the data."
        )
        
        # Filter interesting keys for the prompt to save tokens
        interesting_keys = [
            "acquired", "cloud_cover", "sun_elevation", "sun_azimuth", 
            "ground_control", "view_angle", "satellite_id", "instrument", "gsd"
        ]
        filtered_meta = {k: metadata.get(k, "N/A") for k in interesting_keys}
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Metadata: {json.dumps(filtered_meta)}"}
        ]
        
        payload = {
            "model": self.model, "messages": messages, "temperature": 0.2, "max_tokens": 400
        }
        
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        response = requests.post(AppConfig.GROQ_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        return response.json()["choices"][0]["message"]["content"]

    @staticmethod
    def _extract_json_from_text(text: str) -> Tuple[Optional[Union[Dict, List]], str]:
        if text is None: raise ValueError("Empty text")
        try: return json.loads(text), text
        except: pass
        # Substring search logic
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
                    if not stack: return json.loads(text[start:j+1]), text[start:j+1]
                elif c == ']' and stack and stack[-1] == '[':
                    stack.pop()
                    if not stack: return json.loads(text[start:j+1]), text[start:j+1]
        raise ValueError("No JSON found")

class PlanetAIAgent:
    """Orchestrates conversation, state, and API calls."""
    def __init__(self, llm_api_key: str):
        self.llm = LLMEngine(llm_api_key)
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
        except Exception: return None

    def search_planet_metadata(self, filters: Dict) -> List[Dict]:
        if not AppConfig.PLANET_API_KEY: raise RuntimeError("PLANET_API_KEY not configured")
        
        # Ensure geometry is a dict
        if isinstance(filters.get("geometry"), str):
            g = GeoProcessor.parse_geometry_input(filters["geometry"])
            if g: filters["geometry"] = g
                
        body = self._build_api_body(filters)
        if not body["filter"]["config"]: raise ValueError("No valid filters provided.")
        
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
        if "chat_history" not in st.session_state: st.session_state.chat_history = []
        if "features" in st.session_state: del st.session_state.features
        if "active_preview" in st.session_state: del st.session_state.active_preview

        st.session_state.chat_history.append({"role": "user", "content": user_prompt})

        # Heuristic: Direct Coordinate Detection
        geom_direct = GeoProcessor.parse_geometry_input(user_prompt)
        if geom_direct: st.session_state.assistant_state["geometry"] = geom_direct 

        # PRE-LLM BYPASS (Force Ready)
        state = st.session_state.assistant_state
        if state.get("start_date") and state.get("end_date") and state.get("geometry"):
            assistant_text = "I have the location and dates. Searching Planet's archive now."
            st.session_state.chat_history.append({"role": "assistant", "content": assistant_text})
            filters = {"start_date": state.get("start_date"), "end_date": state.get("end_date"), "cloud_cover": state.get("cloud_cover"), "geometry": state.get("geometry")}
            return {"status": "ready", "assistant_text": assistant_text, "filters": filters}

        # LLM Extraction
        try:
            out = self.llm.extract_filters(user_prompt, st.session_state.chat_history, st.session_state.assistant_state)
        except Exception as e:
            error_msg = f"Brain connection error ({str(e)}). Please retry."
            st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
            return {"status": "error", "assistant_text": error_msg}

        parsed = out["parsed"]
        assistant_text = out["assistant_text"]
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_text})

        # Update State
        state = st.session_state.assistant_state
        for k in ["start_date", "end_date", "cloud_cover", "geometry", "place"]:
            if parsed.get(k): state[k] = parsed.get(k)

        # Geocoding
        if parsed.get("decision") == "defaulted" and not state.get("geometry"):
            place = parsed.get("place") or state.get("place")
            if place:
                geo = self.geocode_place(place)
                if geo and geo.get("lat"):
                    state["geometry"] = GeoProcessor.create_small_bbox_polygon_from_point(geo["lat"], geo["lon"])

        # Check Completion again after LLM update
        is_ready = bool(state.get("start_date") and state.get("end_date") and state.get("geometry"))
        if parsed.get("decision") == "complete" or is_ready:
            filters = {"start_date": state.get("start_date"), "end_date": state.get("end_date"), "cloud_cover": state.get("cloud_cover"), "geometry": state.get("geometry")}
            return {"status": "ready", "assistant_text": assistant_text, "filters": filters}

        return {"status": "need_clarify", "assistant_text": assistant_text, "missing": parsed.get("clarify")}

    def generate_metadata_summary(self, scene_id: str) -> str:
        """Fetches metadata from DB and asks LLM to summarize it."""
        meta = self.db_manager.get_metadata_by_id(scene_id)
        if not meta: return "Metadata not found."
        return self.llm.analyze_metadata(meta)

    def _build_api_body(self, filters: Dict) -> Dict:
        body = {"item_types": ["PSScene"], "filter": {"type": "AndFilter", "config": []}}
        start = self._normalize_date(filters.get("start_date"), "start")
        end = self._normalize_date(filters.get("end_date"), "end")
        if start and end and start > end: start, end = end, start
        
        date_cfg = {}
        if start: date_cfg["gte"] = start
        if end: date_cfg["lte"] = end
        if date_cfg: body["filter"]["config"].append({"type": "DateRangeFilter", "field_name": "acquired", "config": date_cfg})

        cloud = self._normalize_cloud(filters.get("cloud_cover"))
        if cloud is not None: body["filter"]["config"].append({"type": "RangeFilter", "field_name": "cloud_cover", "config": {"lte": cloud}})
        
        geom = filters.get("geometry")
        if geom: body["filter"]["config"].append({"type": "GeometryFilter", "field_name": "geometry", "config": geom})
        return body

    @staticmethod
    def _normalize_date(d: str, mode: str) -> Optional[str]:
        if not d: return None
        s = str(d).strip()
        if "T" in s: return s if s.endswith("Z") else s + "Z"
        m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", s)
        if m:
            y, mo, day = m.groups()
            suffix = "T00:00:00.000Z" if mode == "start" else "T23:59:59.999Z"
            return f"{y}-{int(mo):02d}-{int(day):02d}{suffix}"
        return s

    @staticmethod
    def _normalize_cloud(val: Any) -> Optional[float]:
        if val is None: return None
        try: v = float(val)
        except:
            m = re.search(r"(\d+(\.\d+)?)", str(val))
            if not m: return None
            v = float(m.group(1))
        return v/100.0 if v > 1.0 else v

# --- ORDERS ---
@exponential_backoff_retry(max_retries=3, base_delay=AppConfig.BASE_DELAY)
def place_planet_order(scene_ids: List[str], aoi_geometry: Dict, order_name: str = "Composite Order") -> Dict:
    if not AppConfig.PLANET_API_KEY: return {"success": False, "error": "API Key missing"}
    tools = [{"clip": {"aoi": aoi_geometry}}, {"composite": {}}]
    payload = {
        "name": order_name,
        "source_type": "scenes",
        "products": [{"item_ids": scene_ids, "item_type": "PSScene", "product_bundle": "analytic_udm2"}],
        "tools": tools
    }
    try:
        response = requests.post(AppConfig.PLANET_ORDERS_URL, json=payload, auth=HTTPBasicAuth(AppConfig.PLANET_API_KEY, ""), headers={"Content-Type": "application/json"}, timeout=30)
        if response.status_code in [200, 202]: return {"success": True, "data": response.json()}
        else: return {"success": False, "error": f"{response.status_code}: {response.text}"}
    except Exception as e: return {"success": False, "error": str(e)}

def fetch_thumbnail(url, key):
    try:
        r = requests.get(url, auth=HTTPBasicAuth(key, ""), timeout=30)
        return r.content
    except: return None

# ==================================================================================================
# 7. MAIN UI APPLICATION
# ==================================================================================================

def main():
    AppConfig.validate()
    db = DatabaseManager(AppConfig.DB_PATH)
    agent = PlanetAIAgent(AppConfig.GROQ_API_KEY)
    st.title("🌍 Planet API Assistant")

    if "assistant_state" not in st.session_state:
        st.session_state.assistant_state = {"start_date": None, "end_date": None, "cloud_cover": None, "geometry": None, "place": None}
    if "chat_history" not in st.session_state: st.session_state.chat_history = []

    st.sidebar.title("🛠️ Tools")
    st.sidebar.subheader("1. Area Selection")
    uploaded_file = st.sidebar.file_uploader("Upload Shapefile", type="zip", key="shp_upload")
    
    if uploaded_file and st.session_state.get("last_uploaded_shp") != uploaded_file.name:
        with st.sidebar.status("Processing Shapefile..."):
            geom, err = GeoProcessor.process_uploaded_shapefile(uploaded_file)
            if geom:
                st.session_state.assistant_state["geometry"] = geom
                st.session_state["last_uploaded_shp"] = uploaded_file.name
                st.session_state.chat_history.append({"role": "user", "content": "System Update: I have uploaded a Shapefile."})
                st.session_state.chat_history.append({"role": "assistant", "content": "Received Shapefile. Please provide dates."})
                st.success("Loaded!")
                st.rerun()
            else: st.error(err)

    st.sidebar.markdown("---")
    if st.sidebar.button("🗑️ Start New Chat"):
        for k in list(st.session_state.keys()): del st.session_state[k]
        db.reset_database()
        st.rerun()

    for msg in st.session_state.chat_history:
        if not msg["content"].startswith("System Update"):
            with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if user_text := st.chat_input("Ask..."):
        result = agent.handle_user_prompt(user_text)
        if len(st.session_state.chat_history) >= 2:
            with st.chat_message("user"): st.markdown(st.session_state.chat_history[-2]['content'])
            with st.chat_message("assistant"): st.markdown(st.session_state.chat_history[-1]['content'])
        
        if result.get("status") == "ready":
            filters = result["filters"]
            with st.chat_message("assistant"):
                st.success("✅ Filters set! Searching...")
                with st.expander("Filter Details", expanded=True):
                    st.code(json.dumps(filters, indent=2), language='json')
            
            try:
                with st.spinner("Querying Planet..."):
                    features = agent.search_planet_metadata(filters)
                    st.session_state.features = features
                    if features: st.success(f"Found {len(features)} images.")
                    else: st.warning("No images found.")
            except Exception as e: st.error(f"Search failed: {e}")
        elif result.get("status") == "error": st.error(result.get("assistant_text"))

    if "features" in st.session_state and st.session_state.features:
        features = st.session_state.features
        
        # --- NEW MAP DISPLAY ---
        st.markdown("---")
        st.subheader("🗺️ Coverage Map")
        st.caption("Blue = AOI | Orange = Scenes")
        aoi = st.session_state.assistant_state.get("geometry")
        if aoi:
            map_obj = GeoProcessor.render_search_map(aoi, features)
            if map_obj: st_folium(map_obj, width=1000, height=500)

        # --- ORDERING ---
        st.markdown("---")
        st.subheader("📦 Order & Process")
        with st.container(border=True):
            col1, col2 = st.columns([3, 1])
            with col1:
                ids = [f["id"] for f in features[:50]]
                selected = st.multiselect("Select Scenes:", options=ids)
            with col2:
                st.write(""); st.write("")
                aoi = st.session_state.assistant_state.get("geometry")
                if st.button("🚀 Place Order", disabled=not(selected and aoi), use_container_width=True):
                    with st.spinner("Submitting..."):
                        res = place_planet_order(selected, aoi, f"Order_{int(time.time())}")
                        if res["success"]: st.balloons(); st.success("Ordered!"); st.json(res["data"])
                        else: st.error(res["error"])

        # --- RESULTS TABLE ---
        st.markdown("### Results")
        for f in features[:50]:
            c1, c2, c3, c4, c5 = st.columns([3, 2, 2, 2, 1])
            props = f["properties"]
            c1.write(f["id"]); c2.write(props["acquired"]); c3.write(f"{props['cloud_cover']:.1%}"); c4.write(props["satellite_id"])
            if c5.button("👁️", key=f["id"]):
                t_url = f.get("_links", {}).get("thumbnail")
                if t_url:
                    with st.spinner("Generating AI Analysis from Metadata..."):
                        img = fetch_thumbnail(t_url, AppConfig.PLANET_API_KEY)
                        # NEW: Use Metadata Analyst instead of VLM
                        summary = agent.generate_metadata_summary(f["id"])
                        st.session_state.active_preview = {"id": f["id"], "img": img, "sum": summary}
                        st.rerun()
        
        if "active_preview" in st.session_state:
            p = st.session_state.active_preview
            with st.sidebar:
                st.divider()
                st.image(p["img"], caption=p["id"])
                st.markdown("### 🤖 Analyst Report")
                st.info(p["sum"])

if __name__ == "__main__":
    main()