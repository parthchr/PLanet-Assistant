# planetAgent_enterprise_final.py
"""
Planet API Assistant (Enterprise Edition v6.0 - AWS Ready & UX Optimized)
=========================================================================
A comprehensive AI platform for satellite imagery discovery and analysis.

FEATURES:
1.  **Conversational Interface**: Groq (Llama 3) for intent understanding.
2.  **Multiple Scene Selection**: "Eye" button adds multiple scenes to the side panel.
3.  **Active Highlighting**: Map shows Latest (Lime Green), Selected (Red), and Available (Orange).
4.  **High-Fidelity Shapefiles**: Extracts ~200 coordinates to preserve actual shape boundaries.
5.  **UI & LLM JSON Concealment**: Hides massive geometry arrays from the chat UI and LLM.
6.  **Metadata Analyst (LLM)**: Summarizes database rows (Sun Azimuth, Cloud Cover) into AI reports.
7.  **AWS Deployability**: DB path is env-configurable.

DEPENDENCIES:
    pip install streamlit requests geopandas shapely geopy python-dotenv folium streamlit-folium

USAGE:
    streamlit run planetAgent_enterprise_final.py
"""

import os
import re
import json
import time
import logging
import sqlite3
import zipfile
import tempfile
from typing import Optional, Dict, Any, List, Tuple, Callable, Union

# Third-party imports
import requests
import streamlit as st
import geopandas as gpd
import folium
from streamlit_folium import st_folium
from shapely.geometry import mapping, shape
from shapely.geometry.polygon import orient
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
    PLANET_API_KEY: str = os.getenv("PLANET_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    
    LLM_MODEL: str = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    LLM_TEMP: float = float(os.getenv("LLM_TEMP", "0.3"))
    LLM_MAX_TOKENS: int = 1000
    
    MAX_RETRIES: int = 5
    BASE_DELAY: int = 2
    
    # AWS Deployable: Allow overriding local path for cloud execution environments
    # e.g., set DB_PATH=/tmp/planet.db in AWS Lambda or AppRunner
    DB_PATH: str = os.getenv("DB_PATH", "planet_metadata.db")
    
    PLANET_DATA_URL: str = "https://api.planet.com/data/v1/quick-search"
    GROQ_URL: str = "https://api.groq.com/openai/v1/chat/completions"
    
    @classmethod
    def validate(cls):
        if not cls.PLANET_API_KEY:
            st.error("CRITICAL: PLANET_API_KEY missing from environment variables.")
            st.stop()
        if not cls.GROQ_API_KEY:
            st.warning("WARNING: GROQ_API_KEY missing. Conversational features will fail.")

# ==================================================================================================
# 2. ROBUST RETRY ENGINE
# ==================================================================================================

def exponential_backoff_retry(max_retries: int = 5, base_delay: int = 2):
    """Decorator to handle API Rate Limits (429) with exponential backoff."""
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
    """Manages persistent storage of satellite metadata using SQLite."""
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_schema(self):
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
        conn = self._get_connection()
        c = conn.cursor()
        c.execute("DROP TABLE IF EXISTS metadata")
        conn.commit()
        conn.close()
        self._init_schema()

    def get_metadata_by_id(self, scene_id: str) -> Optional[Dict]:
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM metadata WHERE id = ?", (scene_id,))
        row = c.fetchone()
        conn.close()
        return dict(row) if row else None

    def save_features(self, features: List[Dict[str, Any]]):
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
        Preserves the shape by dynamically simplifying down to ~200 points.
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
                if gdf.crs is not None and gdf.crs.to_string() != "EPSG:4326": 
                    gdf = gdf.to_crs("EPSG:4326")
                if gdf.empty: return None, "Shapefile is empty."
                
                geom = gdf.geometry.iloc[0]
                
                # Ensure we are dealing with a single polygon if it's a MultiPolygon
                if geom.geom_type == 'MultiPolygon':
                    geom = max(geom.geoms, key=lambda a: a.area)
                
                # --- DYNAMIC SIMPLIFICATION ---
                # Reduce complexity until points are <= 200
                tol = 0.0005
                while True:
                    if hasattr(geom, 'exterior'):
                        pt_count = len(geom.exterior.coords)
                        if pt_count <= 200:
                            break
                    else:
                        break # Failsafe
                    geom = geom.simplify(tolerance=tol, preserve_topology=True)
                    tol *= 1.5 
                 
                # RFC 7946 requires the exterior ring to be counter-clockwise.
                # Shapefiles are conventionally clockwise, which Planet's API rejects with a 400.
                geom = orient(geom, sign=1.0)
                return mapping(geom), None

        except Exception as e:
            return None, str(e)

    @staticmethod
    def render_search_map(aoi_geom: Dict, features: List[Dict], selected_ids: List[str] = None, latest_id: str = None):
        """
        Creates a Folium map. 
        - Latest Selection: Lime Green
        - Selected Scenes: Red
        - Unselected Scenes: Orange
        """
        selected_ids = selected_ids or []
        try:
            s = shape(aoi_geom)
            centroid = s.centroid
            m = folium.Map(location=[centroid.y, centroid.x], zoom_start=10, tiles="OpenStreetMap")

            # 1. Add AOI (Blue)
            folium.GeoJson(
                aoi_geom,
                name="Your Area",
                style_function=lambda x: {'color': 'blue', 'fillColor': 'blue', 'fillOpacity': 0.1, 'weight': 2}
            ).add_to(m)

            # Style closure function for dynamic coloring
            def get_style(fid):
                is_latest = (fid == latest_id)
                is_sel = (fid in selected_ids)
                
                if is_latest:
                    return lambda x: {'color': '#39FF14', 'fillColor': '#39FF14', 'fillOpacity': 0.5, 'weight': 4} # LIME GREEN
                elif is_sel:
                    return lambda x: {'color': 'red', 'fillColor': 'red', 'fillOpacity': 0.3, 'weight': 2} # RED
                else:
                    return lambda x: {'color': 'orange', 'fillColor': 'orange', 'fillOpacity': 0.05, 'weight': 1} # ORANGE

            # 2. Add Scenes (Orange/Red/Green)
            for f in features[:50]:
                fid = f.get('id')
                date = f.get('properties', {}).get('acquired')
                
                folium.GeoJson(
                    f['geometry'],
                    name=fid,
                    tooltip=f"ID: {fid}\nDate: {date}",
                    style_function=get_style(fid)
                ).add_to(m)
            
            return m
        except Exception as e:
            st.error(f"Map Error: {e}")
            return None

    @staticmethod
    def parse_geometry_input(value: Any) -> Optional[Dict]:
        if not value: return None
        if isinstance(value, dict): return value
        s = str(value).strip()
        try: return json.loads(s)
        except: pass
        m = re.search(r"\[?\s*(-?\d+(?:\.\d+)?)\s*[,\s]\s*(-?\d+(?:\.\d+)?)\s*[,\s]\s*(-?\d+(?:\.\d+)?)\s*[,\s]\s*(-?\d+(?:\.\d+)?)\s*\]?", s)
        if m:
            coords = [float(x) for x in m.groups()]
            min_lon, min_lat, max_lon, max_lat = coords
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
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = AppConfig.LLM_MODEL
        self.temp = AppConfig.LLM_TEMP

    @exponential_backoff_retry(max_retries=AppConfig.MAX_RETRIES, base_delay=AppConfig.BASE_DELAY)
    def extract_filters(self, user_message: str, recent_history: List[Dict], assistant_state: Dict) -> Dict:
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
        
        # MASK GEOMETRY FROM LLM TO PREVENT TOKEN CRASH
        state_for_llm = assistant_state.copy()
        if state_for_llm.get("geometry"):
            state_for_llm["geometry"] = "[GEOMETRY ALREADY LOADED SAFELY - DO NOT ASK USER FOR LOCATION]"
            
        messages.append({
            "role": "user",
            "content": f"assistant_state = {json.dumps(state_for_llm)}\nuser_message = {user_message}"
        })

        payload = {"model": self.model, "messages": messages, "temperature": self.temp, "max_tokens": 800, "response_format": {"type": "json_object"}}
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        
        response = requests.post(AppConfig.GROQ_URL, headers=headers, json=payload, timeout=40)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]

        try:
            parsed, _ = self._extract_json_from_text(content)
        except:
            parsed = {"decision": "ask", "reply": content}

        reply = parsed.get("reply", "")
        if "{" in reply: reply = re.sub(r'\{.*?\}', '', reply, flags=re.DOTALL)
        parsed["reply"] = reply.strip() or "I've noted that. What else?"
        
        for k in ["start_date", "end_date", "cloud_cover", "geometry", "place"]: parsed.setdefault(k, None)
        if parsed.get("decision") not in ["complete", "ask", "defaulted"]: parsed["decision"] = "ask"
        return {"assistant_text": parsed["reply"], "parsed": parsed}

    @exponential_backoff_retry(max_retries=3, base_delay=2)
    def analyze_metadata(self, metadata: Dict) -> str:
        """Generates a professional summary based on DB metadata columns."""
        system_prompt = (
            "You are a specialized Satellite Imagery Analyst. "
            "Write a concise, professional summary (in bullet points) highlighting key metrics. "
            "Focus on: Acquired Date, Cloud Cover, Sun Elevation, Off-Nadir Angle, and Ground Control. "
            "Make it readable for a human operator deciding if the image is good. "
            "Do NOT hallucinate features not present in the data."
        )
        
        interesting_keys = [
            "acquired", "cloud_cover", "sun_elevation", "sun_azimuth", 
            "ground_control", "view_angle", "satellite_id", "instrument", "gsd"
        ]
        filtered_meta = {k: metadata.get(k, "N/A") for k in interesting_keys}
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Metadata: {json.dumps(filtered_meta)}"}
        ]
        
        payload = {"model": self.model, "messages": messages, "temperature": 0.2, "max_tokens": 400}
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        
        response = requests.post(AppConfig.GROQ_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    @staticmethod
    def _extract_json_from_text(text: str) -> Tuple[Optional[Union[Dict, List]], str]:
        if text is None: raise ValueError("Empty text")
        try: return json.loads(text), text
        except: pass
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
        
        if isinstance(filters.get("geometry"), str):
            g = GeoProcessor.parse_geometry_input(filters["geometry"])
            if g: filters["geometry"] = g
                
        body = self._build_api_body(filters)
        if not body["filter"]["config"]: raise ValueError("No valid filters provided.")
        
        auth = HTTPBasicAuth(AppConfig.PLANET_API_KEY, "")
        headers = {"Content-Type": "application/json"}
        response = requests.post(AppConfig.PLANET_DATA_URL, auth=auth, headers=headers, json=body, timeout=90)
        if not response.ok:
            try:
                detail = response.json()
            except ValueError:
                detail = response.text
            raise RuntimeError(f"Planet API error {response.status_code}: {detail}")
        
        data = response.json()
        features = data.get("features", [])
        self.db_manager.save_features(features)
        return features

    def handle_user_prompt(self, user_prompt: str):
        if "assistant_state" not in st.session_state:
            st.session_state.assistant_state = {"start_date": None, "end_date": None, "cloud_cover": None, "geometry": None, "place": None}
        if "chat_history" not in st.session_state: st.session_state.chat_history = []
        if "features" in st.session_state: del st.session_state.features
        
        # Clear selected previews on new search
        if "selected_previews" in st.session_state: st.session_state.selected_previews = {}
        if "last_selected_scene" in st.session_state: st.session_state.last_selected_scene = None

        st.session_state.chat_history.append({"role": "user", "content": user_prompt})
        geom_direct = GeoProcessor.parse_geometry_input(user_prompt)
        if geom_direct: st.session_state.assistant_state["geometry"] = geom_direct 

        # PRE-LLM BYPASS (Force Ready Check)
        state = st.session_state.assistant_state
        if state.get("start_date") and state.get("end_date") and state.get("geometry"):
            assistant_text = "I have the location and dates. Searching Planet's archive now."
            st.session_state.chat_history.append({"role": "assistant", "content": assistant_text})
            filters = {"start_date": state.get("start_date"), "end_date": state.get("end_date"), "cloud_cover": state.get("cloud_cover"), "geometry": state.get("geometry")}
            return {"status": "ready", "assistant_text": assistant_text, "filters": filters}

        try:
            out = self.llm.extract_filters(user_prompt, st.session_state.chat_history, st.session_state.assistant_state)
        except Exception as e:
            error_msg = f"Brain error: {str(e)}"
            st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
            return {"status": "error", "assistant_text": error_msg}

        parsed = out["parsed"]
        assistant_text = out["assistant_text"]
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_text})

        # State Override Protection
        state = st.session_state.assistant_state
        for k in ["start_date", "end_date", "cloud_cover", "geometry", "place"]:
            val = parsed.get(k)
            if val:
                if k == "geometry" and isinstance(val, str) and "LOADED SAFELY" in val:
                    continue
                elif k == "geometry" and isinstance(val, str):
                    parsed_geom = GeoProcessor.parse_geometry_input(val)
                    if parsed_geom: state[k] = parsed_geom
                else:
                    state[k] = val

        if parsed.get("decision") == "defaulted" and not state.get("geometry"):
            place = parsed.get("place") or state.get("place")
            if place:
                geo = self.geocode_place(place)
                if geo and geo.get("lat"):
                    half_km = sqrt(30) / 2
                    state["geometry"] = GeoProcessor.create_small_bbox_polygon_from_point(geo["lat"], geo["lon"], half_km)

        is_ready = bool(state.get("start_date") and state.get("end_date") and state.get("geometry"))
        if parsed.get("decision") == "complete" or is_ready:
            filters = {"start_date": state.get("start_date"), "end_date": state.get("end_date"), "cloud_cover": state.get("cloud_cover"), "geometry": state.get("geometry")}
            return {"status": "ready", "assistant_text": assistant_text, "filters": filters}

        return {"status": "need_clarify", "assistant_text": assistant_text, "missing": parsed.get("clarify")}

    def generate_metadata_summary(self, scene_id: str) -> str:
        meta = self.db_manager.get_metadata_by_id(scene_id)
        if not meta: return "Metadata not found in Database."
        return self.llm.analyze_metadata(meta)

    def _build_api_body(self, filters: Dict) -> Dict:
        body = {"item_types": ["PSScene"], "filter": {"type": "AndFilter", "config": []}}
        start = _normalize_date_iso(filters.get("start_date"), "start")
        end = _normalize_date_iso(filters.get("end_date"), "end")
        if start and end and start > end: start, end = end, start
        
        date_config = {}
        if start: date_config["gte"] = start
        if end: date_config["lte"] = end
        if date_config: body["filter"]["config"].append({"type": "DateRangeFilter", "field_name": "acquired", "config": date_config})

        cloud = _normalize_cloud_cover(filters.get("cloud_cover"))
        if cloud is not None: body["filter"]["config"].append({"type": "RangeFilter", "field_name": "cloud_cover", "config": {"lte": cloud}})
        geom = filters.get("geometry")
        if geom: body["filter"]["config"].append({"type": "GeometryFilter", "field_name": "geometry", "config": geom})
        return body

# --- UTILS ---
def fetch_thumbnail(url, key):
    try:
        r = requests.post(url, auth=HTTPBasicAuth(key, ""), timeout=30)
        r.raise_for_status()
        return r.content
    except Exception: return None

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
    try: v = float(val)
    except:
        m = re.search(r"(\d+(\.\d+)?)", str(val))
        if not m: return None
        v = float(m.group(1))
    return v/100.0 if v > 1.0 else v

# ==================================================================================================
# 7. MAIN UI APPLICATION
# ==================================================================================================

def main():
    AppConfig.validate()
    db = DatabaseManager(AppConfig.DB_PATH)
    agent = PlanetAIAgent(AppConfig.GROQ_API_KEY)
    
    st.markdown("""
        <div style='border-bottom: 1px solid #4a4a4a; padding-bottom: 10px; margin-bottom: 20px;'>
            <h1 style='margin-bottom: 0px;'>🌍 Planet API Assistant</h1>
            <p style='color: #888; font-size: 1.1em; margin-top: 5px;'>Enterprise Satellite Data Discovery Chatbot</p>
        </div>
    """, unsafe_allow_html=True)

    if "assistant_state" not in st.session_state:
        st.session_state.assistant_state = {"start_date": None, "end_date": None, "cloud_cover": None, "geometry": None, "place": None}
    if "chat_history" not in st.session_state: st.session_state.chat_history = []
    
    # State for multiple previews and latest selection
    if "selected_previews" not in st.session_state: st.session_state.selected_previews = {}
    if "last_selected_scene" not in st.session_state: st.session_state.last_selected_scene = None

    st.sidebar.title("🛠️ Tools")
    st.sidebar.subheader("1. Area Selection")
    uploaded_file = st.sidebar.file_uploader("Upload Shapefile (.zip)", type="zip", key="shp_upload")
    
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

    # --- SHOW SELECTED PREVIEWS IN SIDEBAR ---
    if st.session_state.selected_previews:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Analysis Reports")
        
        # Display the latest one dynamically
        for sid, pdata in st.session_state.selected_previews.items():
            is_latest = (sid == st.session_state.last_selected_scene)
            
            expander_title = f"⭐ LATEST: {sid}" if is_latest else f"Scene: {sid}"
            
            with st.sidebar.expander(expander_title, expanded=is_latest):
                if is_latest:
                    st.success("Currently Active Selection on Map")
                if pdata["img"]:
                    st.image(pdata["img"], use_container_width=True)
                st.markdown(pdata["sum"])

    # --- CHAT UI ---
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
                    # Hide giant geometry from UI if it's from a shapefile
                    display_filters = filters.copy()
                    if st.session_state.get("last_uploaded_shp"):
                        display_filters["geometry"] = "[Hidden for display - Shapefile Geometry Loaded]"
                    st.code(json.dumps(display_filters, indent=2), language='json')
            
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
        
        # --- MAP DISPLAY ---
        st.markdown("---")
        st.subheader("🗺️ Coverage Map")
        st.caption("Blue = AOI | Orange = Available | Red = Selected | Lime Green = Latest Selection")
        aoi = st.session_state.assistant_state.get("geometry")
        if aoi:
            sel_ids = list(st.session_state.selected_previews.keys())
            latest_id = st.session_state.last_selected_scene
            
            map_obj = GeoProcessor.render_search_map(aoi, features, sel_ids, latest_id)
            if map_obj: 
                st_folium(map_obj, width=1000, height=500, returned_objects=[])

        # --- RESULTS TABLE ---
        st.markdown("---")
        st.markdown("### Search Results")
        st.caption("Click 'Add to Analysis' to summarize metadata and highlight the scene on the map.")
        
        preview_list = []
        for f in features[:50]:
            props = f.get("properties", {})
            links = f.get("_links", {})
            preview_list.append({
                "id": f.get("id"), "acquired": props.get("acquired"), 
                "cloud": props.get("cloud_cover"), "satellite": props.get("satellite_id") or props.get("item_type"), 
                "thumbnail": links.get("thumbnail")
            })
        
        h1, h2, h3, h4, h5 = st.columns([3, 2, 2, 2, 2])
        h1.markdown("**Scene ID**"); h2.markdown("**Date**"); h3.markdown("**Cloud Cover**"); h4.markdown("**Satellite**"); h5.markdown("**Action**")
        
        for item in preview_list:
            c1, c2, c3, c4, c5 = st.columns([3, 2, 2, 2, 2])
            c1.write(item["id"])
            c2.write(item["acquired"])
            c3.write(f"{item['cloud']:.2%}" if item['cloud'] is not None else "N/A")
            c4.write(item["satellite"])
            
            # Multi-select Toggle Button
            is_selected = item["id"] in st.session_state.selected_previews
            btn_label = "➖ Remove" if is_selected else "👁️ Add to Analysis"
            
            if c5.button(btn_label, key=f"btn_{item['id']}"):
                if is_selected:
                    del st.session_state.selected_previews[item["id"]]
                    # If we removed the latest scene, reset the latest tracker
                    if st.session_state.last_selected_scene == item["id"]:
                        st.session_state.last_selected_scene = None
                    st.rerun()
                else:
                    if item["thumbnail"]:
                        with st.spinner(f"Generating Report for {item['id']}..."):
                            img = fetch_thumbnail(item["thumbnail"], AppConfig.PLANET_API_KEY)
                            summary = agent.generate_metadata_summary(item["id"])
                            st.session_state.selected_previews[item["id"]] = {"img": img, "sum": summary}
                            
                            # Track this as the most recent selection
                            st.session_state.last_selected_scene = item["id"]
                            st.rerun()
                    else: 
                        st.warning("No thumbnail.")

if __name__ == "__main__":
    main()