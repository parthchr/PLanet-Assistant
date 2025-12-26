# planetAgent_final.py
"""
Planet API Assistant (COMPLETED INTEGRATED VERSION)

Features:
  - ORIGINAL: Humanlike/Receptionist Chat (Groq/Llama3)
  - ORIGINAL: Full Metadata Database Storage (SQLite)
  - ORIGINAL: Planet API Search
  - NEW: VLM Image Preview & Summarization (Ollama/LLaVA)
  - NEW: Shapefile Upload (.zip) for AOI
  - NEW: Image Clipping (Crops thumbnail to your AOI)

Usage:
  - Put PLANET_API_KEY and GROQ_API_KEY in .env
  - Run: streamlit run planetAgent_final.py
  - REQUIRES OLLAMA to be running locally with 'llava' model pulled.
"""
import os
import re
import json
import sqlite3
import requests
import streamlit as st
from geopy.geocoders import Nominatim
from math import cos, radians, sqrt
from datetime import datetime, timedelta
from dotenv import load_dotenv
import base64
import io
from requests.auth import HTTPBasicAuth
import zipfile
import tempfile
import geopandas as gpd
from shapely.geometry import shape, box
from PIL import Image

# ---------- Load env ----------
load_dotenv()
PLANET_API_KEY = os.getenv("PLANET_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
try:
    LLM_TEMP = float(os.getenv("LLM_TEMP", "0.3"))
except Exception:
    LLM_TEMP = 0.3

st.set_page_config(page_title="Planet Assistant", layout="wide")

# ---------- DB (ORIGINAL COMPLEX SCHEMA) ----------
DB_PATH = "planet_metadata.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
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

def reset_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DROP TABLE IF EXISTS metadata")
    conn.commit()
    conn.close()
    init_db()

def save_metadata_to_db(metadata_list):
    if not metadata_list: return
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    cols = [
        "id","item_type","acquired","anomalous_pixels","clear_confidence_percent",
        "clear_percent","cloud_cover","cloud_percent","ground_control","gsd",
        "heavy_haze_percent","instrument","pixel_resolution","provider",
        "published","publishing_stage","quality_category","satellite_azimuth",
        "satellite_id","shadow_percent","snow_ice_percent","strip_id",
        "sun_azimuth","sun_elevation","updated","view_angle","visible_confidence_percent",
        "visible_percent","geometry","full_metadata"
    ]
    placeholders = ",".join(["?"]*len(cols))
    sql = f"INSERT OR REPLACE INTO metadata ({','.join(cols)}) VALUES ({placeholders})"
    for item in metadata_list:
        p = item.get("properties", {}) or {}
        geom = item.get("geometry")
        row = (
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
            json.dumps(geom) if geom is not None else None,
            json.dumps(item)
        )
        try:
            c.execute(sql, row)
        except Exception as e:
            continue
    conn.commit()
    conn.close()

# ---------- Helpers ----------
def _normalize_date_iso(date_str, which="start"):
    if not date_str: return None
    s = str(date_str).strip()
    if "T" in s: return s if s.endswith("Z") else s + "Z"
    m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", s)
    if m:
        if which == "start": return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}T00:00:00.000Z"
        else: return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}T23:59:59.999Z"
    return s

def _normalize_cloud_cover(val):
    if val is None: return None
    try: v = float(val)
    except Exception:
        m = re.search(r"(\d+(\.\d+)?)", str(val))
        if not m: return None
        v = float(m.group(1))
    if v > 1: v = v/100.0
    return max(0.0, min(1.0, v))

def parse_geometry_input(value):
    if not value: return None
    if isinstance(value, dict):
        if value.get("type") and value.get("coordinates"): return value
        return None
    s = str(value).strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and obj.get("type") and obj.get("coordinates"): return obj
    except Exception: pass
    m = re.search(r"\[?\s*(-?\d+(?:\.\d+)?)\s*[,\s]\s*(-?\d+(?:\.\d+)?)\s*[,\s]\s*(-?\d+(?:\.\d+)?)\s*[,\s]\s*(-?\d+(?:\.\d+)?)\s*\]?", s)
    if m:
        min_lon, min_lat, max_lon, max_lat = map(float, (m.group(1), m.group(2), m.group(3), m.group(4)))
        if min_lon > max_lon: min_lon, max_lon = max_lon, min_lon
        if min_lat > max_lat: min_lat, max_lat = max_lat, min_lat
        coords = [[min_lon, min_lat],[max_lon, min_lat],[max_lon, max_lat],[min_lon, max_lat],[min_lon, min_lat]]
        return {"type":"Polygon","coordinates":[coords]}
    return None

def create_small_bbox_polygon_from_point(lat, lon, half_km=2.74):
    deg_lat = half_km / 111.0
    deg_lon = half_km / (111.0 * abs(cos(radians(lat)) or 1.0))
    min_lon = lon - deg_lon; max_lon = lon + deg_lon
    min_lat = lat - deg_lat; max_lat = lat + deg_lat
    coords = [[min_lon, min_lat],[max_lon, min_lat],[max_lon, max_lat],[min_lon, max_lat],[min_lon, min_lat]]
    return {"type":"Polygon","coordinates":[coords]}

# ### <<< NEW FEATURE: Shapefile Handling >>>
def handle_shapefile_upload(uploaded_file):
    """
    Extracts geometry from an uploaded zip file (Shapefile).
    Returns GeoJSON dict (Polygon/MultiPolygon).
    """
    try:
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Save and unzip
            zip_path = os.path.join(tmpdirname, "uploaded.zip")
            with open(zip_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
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
                return None, "No .shp file found in the zip."
            
            # Read with geopandas
            gdf = gpd.read_file(shp_file)
            if gdf.crs != "EPSG:4326":
                gdf = gdf.to_crs("EPSG:4326")
            
            # Convert to GeoJSON geometry
            # Using the convex hull of all shapes combined to get a single search geometry
            combined = gdf.unary_union
            geom_json = json.loads(json.dumps(combined.__geo_interface__))
            return geom_json, None
            
    except Exception as e:
        return None, str(e)

# ---------- Robust JSON extraction ----------
def find_first_json_substring(text: str):
    if not text or not isinstance(text, str): return None
    length = len(text)
    for i,ch in enumerate(text):
        if ch not in ('{','['): continue
        start = i; stack=[ch]; in_str=False; esc=False
        for j in range(i+1,length):
            c = text[j]
            if esc: esc=False; continue
            if c == '\\': esc=True; continue
            if c == '"' and not esc: in_str = not in_str; continue
            if in_str: continue
            if c == '{': stack.append('{')
            elif c == '[': stack.append('[')
            elif c == '}' and stack and stack[-1] == '{':
                stack.pop()
                if not stack: return text[start:j+1]
            elif c == ']' and stack and stack[-1] == '[':
                stack.pop()
                if not stack: return text[start:j+1]
    return None

def extract_json_from_text(text: str):
    if text is None: raise ValueError("empty text")
    try:
        parsed = json.loads(text); return parsed, text
    except Exception:
        pass
    substr = find_first_json_substring(text)
    if not substr: raise ValueError("No JSON found in LLM response.")
    parsed = json.loads(substr)
    return parsed, substr

# ---------- LLM Extractor ----------
class LLMExtractor:
    def __init__(self, api_key, model=LLM_MODEL, temp=LLM_TEMP):
        self.api_key = api_key
        self.model = model
        self.temp = temp

    def extract_and_reply(self, user_message: str, recent_history: list, assistant_state: dict):
        if not self.api_key: raise RuntimeError("LLM API key not configured")
        
        system_prompt = (
            "You are a warm, human receptionist-style assistant. Your job: TALK naturally and politely with the user, "
            "and collect four filters: start_date, end_date, cloud_cover, geometry (bbox or GeoJSON). "
            "If the user says they uploaded a shapefile, trust that 'geometry' is handled and move to other filters. "
            "If data is missing, politely ask for just that missing information. "
            "Output JSON keys: start_date, end_date, cloud_cover, geometry, place, decision (complete/ask/defaulted), reply."
        )

        messages = [{"role":"system","content":system_prompt}]
        for h in (recent_history or [])[-6:]: messages.append(h)
        messages.append({"role":"user","content":f"assistant_state = {json.dumps(assistant_state)}\n\nuser_message = {json.dumps(user_message)}"})

        payload = {"model": self.model, "messages": messages, "temperature": self.temp, "max_tokens": 600}
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type":"application/json"}
        r = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=40)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]

        parsed = None
        try:
            parsed, substr = extract_json_from_text(text)
            if isinstance(parsed, list): parsed = parsed[0]
        except Exception:
            parsed = {"start_date":None,"end_date":None,"cloud_cover":None,"geometry":None,"place":None,"decision":"ask","reply":None}
            dates = re.findall(r"\d{4}-\d{1,2}-\d{1,2}", text)
            if dates:
                parsed["start_date"] = dates[0]
                if len(dates) > 1: parsed["end_date"] = dates[1]
            parsed["reply"] = text.strip()[:400]

        assistant_text = parsed.get("reply") or parsed.get("reasoning")
        if not assistant_text: assistant_text = text.strip()

        for k in ["start_date","end_date","cloud_cover","geometry","place","decision","clarify","reply","reasoning"]:
            parsed.setdefault(k, None)
        if parsed.get("decision") not in ("complete","ask","defaulted"): parsed["decision"] = "ask"
        if not parsed.get("reply"): parsed["reply"] = assistant_text
        return {"assistant_text": assistant_text, "parsed": parsed}

# ---------- Planet Agent ----------
class PlanetAIAgent:
    def __init__(self, llm_api_key):
        self.llm = LLMExtractor(llm_api_key)
        self.geolocator = Nominatim(user_agent="planet-assistant")

    def geocode_place(self, place_name):
        try:
            loc = self.geolocator.geocode(place_name, addressdetails=True)
            if not loc: return None
            raw = getattr(loc, "raw", {})
            bbox = None
            if "boundingbox" in raw:
                try:
                    south = float(raw["boundingbox"][0]); north = float(raw["boundingbox"][1])
                    west = float(raw["boundingbox"][2]); east = float(raw["boundingbox"][3])
                    bbox = (south, west, north, east)
                except Exception: bbox = None
            return {"lat": loc.latitude, "lon": loc.longitude, "bbox": bbox, "display_name": raw.get("display_name")}
        except Exception: return None

    def search_planet_metadata(self, filters: dict):
        if not PLANET_API_KEY: raise RuntimeError("PLANET_API_KEY not configured")
        if isinstance(filters.get("geometry"), str):
            g = parse_geometry_input(filters["geometry"])
            if g: filters["geometry"] = g
        body = build_planet_api_body(filters)
        if not body["filter"]["config"]: raise ValueError("No valid filters to search.")
        url = "https://api.planet.com/data/v1/quick-search"
        auth = (PLANET_API_KEY, "")
        headers = {"Content-Type":"application/json"}
        r = requests.post(url, auth=auth, headers=headers, json=body, timeout=90)
        r.raise_for_status()
        data = r.json()
        features = data.get("features", [])
        save_metadata_to_db(features)
        return features

    def handle_user_prompt(self, user_prompt: str):
        if "assistant_state" not in st.session_state:
            st.session_state.assistant_state = {"start_date":None,"end_date":None,"cloud_cover":None,"geometry":None,"place":None}
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        # Clear old results
        if "features" in st.session_state: del st.session_state.features
        if "active_preview" in st.session_state: del st.session_state.active_preview

        st.session_state.chat_history.append({"role":"user","content":user_prompt})

        # Check for uploaded shapefile geometry in session state (set by UI)
        if "shapefile_geometry" in st.session_state and st.session_state.shapefile_geometry:
             st.session_state.assistant_state["geometry"] = st.session_state.shapefile_geometry

        geom_direct = parse_geometry_input(user_prompt)
        if geom_direct:
            st.session_state.assistant_state["geometry"] = geom_direct

        try:
            out = self.llm.extract_and_reply(user_prompt, st.session_state.chat_history, st.session_state.assistant_state)
        except Exception as e:
            assistant_text = f"LLM error: {e}"
            st.session_state.chat_history.append({"role":"assistant","content":assistant_text})
            return {"status":"error","assistant_text":assistant_text}

        assistant_text = out["assistant_text"]
        parsed = out["parsed"]

        st.session_state.chat_history.append({"role":"assistant","content":assistant_text})

        state = st.session_state.assistant_state
        if parsed.get("start_date"): state["start_date"] = parsed.get("start_date")
        if parsed.get("end_date"): state["end_date"] = parsed.get("end_date")
        if parsed.get("cloud_cover"): state["cloud_cover"] = parsed.get("cloud_cover")
        if parsed.get("place"): state["place"] = parsed.get("place")
        if parsed.get("geometry"):
            gp = parsed.get("geometry")
            gp_parsed = parse_geometry_input(gp) if isinstance(gp, str) else gp
            if gp_parsed: state["geometry"] = gp_parsed

        # Assume Logic
        user_low = user_prompt.lower()
        if any(p in user_low for p in ["assume", "don't have coordinates"]) and not state.get("geometry"):
            place = state.get("place")
            if place:
                geo = self.geocode_place(place)
                if geo:
                    state["geometry"] = create_small_bbox_polygon_from_point(geo["lat"], geo["lon"])

        if parsed.get("decision") == "complete":
            filters = {
                "start_date": state.get("start_date"),
                "end_date": state.get("end_date"),
                "cloud_cover": state.get("cloud_cover"),
                "geometry": state.get("geometry")
            }
            return {"status":"ready","assistant_text":assistant_text,"filters":filters}

        return {"status":"need_clarify","assistant_text":assistant_text,"missing": parsed.get("clarify")}

def build_planet_api_body(filters: dict):
    body = {"item_types":["PSScene"], "filter":{"type":"AndFilter","config":[]}}
    start = _normalize_date_iso(filters.get("start_date"), "start")
    end = _normalize_date_iso(filters.get("end_date"), "end")
    if start and end and start > end: start, end = end, start
    
    date_config = {}
    if start: date_config["gte"] = start
    if end: date_config["lte"] = end
    if date_config:
        body["filter"]["config"].append({"type": "DateRangeFilter","field_name": "acquired","config": date_config})

    cloud = _normalize_cloud_cover(filters.get("cloud_cover"))
    if cloud is not None:
        body["filter"]["config"].append({"type":"RangeFilter","field_name":"cloud_cover","config":{"lte":cloud}})
    
    geom = filters.get("geometry")
    if isinstance(geom, dict):
        body["filter"]["config"].append({"type":"GeometryFilter","field_name":"geometry","config":geom})
    return body

# ### <<< NEW FEATURE: Clipping Logic >>>
def clip_image_to_geometry(image_bytes, image_geometry, clip_geometry):
    """
    Crops the image (which covers image_geometry) to the clip_geometry (user AOI).
    This is an approximation using bounding boxes for speed on standard thumbnails.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        
        # Get bounds of the full satellite image
        img_shape = shape(image_geometry)
        minx, miny, maxx, maxy = img_shape.bounds
        
        # Get bounds of the clip geometry (user's AOI)
        clip_shape = shape(clip_geometry)
        c_minx, c_miny, c_maxx, c_maxy = clip_shape.bounds

        # Calculate pixel dimensions
        width, height = img.size
        
        # Calculate scaling factors
        x_scale = width / (maxx - minx)
        y_scale = height / (maxy - miny)
        
        # Calculate pixel coordinates for the crop
        # Note: Latitude (y) is usually inverted in pixel coords (0 at top) vs Map coords (0 at equator)
        # But for simple bbox crop on un-projected thumbnail, we map relative positions.
        
        left = int((c_minx - minx) * x_scale)
        right = int((c_maxx - minx) * x_scale)
        # Y is inverted: maxy matches pixel 0
        top = int((maxy - c_maxy) * y_scale)
        bottom = int((maxy - c_miny) * y_scale)
        
        # Clamp values
        left = max(0, left)
        top = max(0, top)
        right = min(width, right)
        bottom = min(height, bottom)
        
        if right <= left or bottom <= top:
            return image_bytes # Return original if crop is invalid
            
        cropped_img = img.crop((left, top, right, bottom))
        
        # Convert back to bytes
        buf = io.BytesIO()
        cropped_img.save(buf, format="PNG")
        return buf.getvalue()

    except Exception as e:
        print(f"Clipping error: {e}")
        return image_bytes # Return original on failure

def fetch_thumbnail(thumbnail_url, api_key):
    try:
        auth = HTTPBasicAuth(api_key, "")
        response = requests.get(thumbnail_url, auth=auth, timeout=30)
        response.raise_for_status()
        return response.content
    except Exception as e:
        print(f"Error fetching thumbnail: {e}")
        return None

def get_vlm_summary(image_bytes):
    try:
        encoded_image = base64.b64encode(image_bytes).decode('utf-8')
        ollama_url = "http://localhost:11434/api/generate"
        payload = {"model": "llava", "prompt": "Describe this satellite image in a single, concise paragraph.", "images": [encoded_image], "stream": False}
        response = requests.post(ollama_url, json=payload, timeout=60)
        response.raise_for_status()
        return response.json().get("response", "No summary").strip()
    except Exception as e:
        return f"VLM Error: {e}"

# ---------- Streamlit UI ----------
def main():
    st.title("🌍 Planet API Assistant (Integrated w/ Shapefiles & Clipping)")

    # Sidebar
    st.sidebar.markdown("## Controls")
    if st.sidebar.button("Start New Chat"):
        for k in list(st.session_state.keys()): del st.session_state[k]
        reset_db()
        st.rerun()
    
    # ### <<< NEW FEATURE: Shapefile Upload in Sidebar >>>
    st.sidebar.markdown("### 1. Upload Area (Optional)")
    uploaded_shp = st.sidebar.file_uploader("Upload .zip Shapefile", type="zip")
    if uploaded_shp:
        if "shapefile_geometry" not in st.session_state:
            geom, err = handle_shapefile_upload(uploaded_shp)
            if geom:
                st.session_state.shapefile_geometry = geom
                if "assistant_state" not in st.session_state: st.session_state.assistant_state = {}
                st.session_state.assistant_state["geometry"] = geom
                st.sidebar.success("✅ Shapefile loaded! Tell the chat to search.")
            else:
                st.sidebar.error(f"Shapefile Error: {err}")

    init_db()
    agent = PlanetAIAgent(GROQ_API_KEY)
    
    if "chat_history" not in st.session_state: st.session_state.chat_history = []

    # Chat Interface
    for msg in st.session_state.chat_history:
        st.chat_message(msg["role"]).write(msg["content"])

    if user_text := st.chat_input("Ask about imagery..."):
        result = agent.handle_user_prompt(user_text)
        st.rerun()

    # Results Display
    # Logic to show results if they exist (handling the rerun flow)
    # Re-instating the logic from previous steps to display assistant response just after input
    
    if "features" in st.session_state:
        features = st.session_state.features
        st.markdown(f"### Results ({len(features)})")
        
        # Table Header
        c1, c2, c3, c4 = st.columns([3, 2, 2, 1])
        c1.write("**ID**"); c2.write("**Date**"); c3.write("**Cloud**"); c4.write("**Action**")
        
        for f in features[:50]:
            col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
            col1.write(f["id"])
            col2.write(f["properties"]["acquired"])
            col3.write(f["properties"]["cloud_cover"])
            
            # Eye Button
            if col4.button("👁️", key=f["id"]):
                t_url = f["_links"]["thumbnail"]
                if t_url:
                    with st.spinner("Fetching, Clipping & Analyzing..."):
                        # 1. Fetch
                        raw_bytes = fetch_thumbnail(t_url, PLANET_API_KEY)
                        
                        if raw_bytes:
                            # 2. Clip (NEW FEATURE)
                            final_image = raw_bytes
                            user_geom = st.session_state.assistant_state.get("geometry")
                            feat_geom = f["geometry"]
                            
                            if user_geom:
                                final_image = clip_image_to_geometry(raw_bytes, feat_geom, user_geom)
                            
                            # 3. VLM
                            summary = get_vlm_summary(final_image)
                            
                            st.session_state.active_preview = {
                                "id": f["id"],
                                "image": final_image,
                                "summary": summary
                            }
                            st.rerun()
                else:
                    st.error("No thumbnail link.")

        # Active Preview
        if "active_preview" in st.session_state:
            p = st.session_state.active_preview
            with st.expander(f"Analysis: {p['id']}", expanded=True):
                st.image(p["image"], caption="Clipped Thumbnail")
                st.info(p["summary"])

if __name__ == "__main__":
    main()