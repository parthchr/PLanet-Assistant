# planetAgent_deployed.py
"""
Planet API Assistant (FULL INTEGRATED DEPLOYMENT VERSION)

Features:
  - SMART MEMORY: LLM sees current state + shapefile uploads immediately.
  - UI: Upload button moved to main chat area (above input).
  - DB: Hybrid SQLite (Local) / PostgreSQL (Cloud) with FULL SCHEMA.
  - VLM: Hugging Face (BLIP) for image analysis.
  - GEO: Shapefile support + Image Clipping.

Usage:
  - Local: .env with PLANET_API_KEY, GROQ_API_KEY, HF_TOKEN
  - Cloud: Environment variables in Render/Heroku
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

# Postgres import check for Cloud
try:
    import psycopg2
except ImportError:
    psycopg2 = None

# ---------- Load env ----------
load_dotenv()
PLANET_API_KEY = os.getenv("PLANET_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN") 
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
try:
    LLM_TEMP = float(os.getenv("LLM_TEMP", "0.3"))
except Exception:
    LLM_TEMP = 0.3

st.set_page_config(page_title="Planet Assistant", layout="wide")

# ---------- DB (HYBRID & FULL SCHEMA) ----------
DB_PATH = "planet_metadata.db"

def get_db_connection():
    db_url = os.getenv("DATABASE_URL")
    if db_url and psycopg2:
        return psycopg2.connect(db_url, sslmode='require')
    return sqlite3.connect(DB_PATH)

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    
    # RESTORED FULL SCHEMA
    query = """
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
    """
    c.execute(query)
    conn.commit()
    conn.close()

def reset_db():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("DROP TABLE IF EXISTS metadata")
    conn.commit()
    conn.close()
    init_db()

def save_metadata_to_db(metadata_list):
    if not metadata_list: return
    conn = get_db_connection()
    c = conn.cursor()
    
    # RESTORED FULL COLUMN MAPPING
    cols = [
        "id","item_type","acquired","anomalous_pixels","clear_confidence_percent",
        "clear_percent","cloud_cover","cloud_percent","ground_control","gsd",
        "heavy_haze_percent","instrument","pixel_resolution","provider",
        "published","publishing_stage","quality_category","satellite_azimuth",
        "satellite_id","shadow_percent","snow_ice_percent","strip_id",
        "sun_azimuth","sun_elevation","updated","view_angle","visible_confidence_percent",
        "visible_percent","geometry","full_metadata"
    ]
    
    is_postgres = bool(os.getenv("DATABASE_URL"))
    ph = "%s" if is_postgres else "?"
    placeholders = ",".join([ph]*len(cols))
    
    sql = f"INSERT INTO metadata ({','.join(cols)}) VALUES ({placeholders}) ON CONFLICT (id) DO NOTHING"
    if not is_postgres:
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
            json.dumps(geom) if geom else None,
            json.dumps(item)
        )
        try:
            c.execute(sql, row)
        except Exception: continue
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
    if isinstance(value, dict): return value
    s = str(value).strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and obj.get("type"): return obj
    except: pass
    m = re.search(r"\[?\s*(-?\d+\.?\d*)\s*[,\s]\s*(-?\d+\.?\d*)\s*[,\s]\s*(-?\d+\.?\d*)\s*[,\s]\s*(-?\d+\.?\d*)\s*\]?", s)
    if m:
        v = list(map(float, m.groups()))
        min_lon, min_lat, max_lon, max_lat = v[0], v[1], v[2], v[3]
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

def handle_shapefile_upload(uploaded_file):
    try:
        with tempfile.TemporaryDirectory() as tmpdirname:
            zip_path = os.path.join(tmpdirname, "uploaded.zip")
            with open(zip_path, "wb") as f: f.write(uploaded_file.getbuffer())
            with zipfile.ZipFile(zip_path, 'r') as z: z.extractall(tmpdirname)
            
            shp_file = next((os.path.join(r, f) for r, d, fs in os.walk(tmpdirname) for f in fs if f.endswith(".shp")), None)
            if not shp_file: return None, "No .shp found"
            
            gdf = gpd.read_file(shp_file)
            if gdf.crs != "EPSG:4326": gdf = gdf.to_crs("EPSG:4326")
            
            combined = gdf.unary_union
            return json.loads(json.dumps(combined.__geo_interface__)), None
    except Exception as e: return None, str(e)

# ---------- BRAIN: LLM EXTRACTOR ----------
def extract_json_from_text(text):
    try:
        start = text.find('{')
        end = text.rfind('}') + 1
        if start != -1 and end != 0: return json.loads(text[start:end]), text
        return json.loads(text), text
    except: return {}, text

class LLMExtractor:
    def __init__(self, api_key, model=LLM_MODEL):
        self.api_key = api_key
        self.model = model

    def extract_and_reply(self, user_message, history, state):
        if not self.api_key: return {"assistant_text": "Error: Key missing", "parsed":{}}

        # SYSTEM PROMPT (Fixed Memory)
        system_prompt = (
            "You are a smart satellite imagery assistant. Collect 4 filters: "
            "start_date, end_date, cloud_cover, geometry.\n"
            "CURRENT STATE: " + json.dumps(state) + "\n"
            "RULES:\n"
            "1. IF 'geometry' is in CURRENT STATE, DO NOT ask for it. Assume it is set.\n"
            "2. If user mentions 'shapefile' or 'uploaded', assume geometry is handled.\n"
            "3. Extract cloud cover from phrases like 'less than 0.25' or '10%'.\n"
            "4. Only ask for missing fields. If all 4 exist, set 'decision'='complete'.\n"
            "5. Output JSON with keys: start_date, end_date, cloud_cover, geometry, place, decision, reply."
        )

        messages = [{"role":"system","content":system_prompt}] + (history or [])[-6:]
        messages.append({"role":"user","content": f"User Input: {user_message}"})

        try:
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type":"application/json"}
            payload = {"model": self.model, "messages": messages, "temperature": 0.3}
            r = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=30)
            text = r.json()["choices"][0]["message"]["content"]
            
            parsed, _ = extract_json_from_text(text)
            
            # Regex Fallback for Cloud Cover
            if not parsed.get("cloud_cover"):
                m = re.search(r"(?:cloud|cover).*?([<>=]?\s*\d+(?:\.\d+)?)\s*(%|percent)?", user_message, re.I)
                if m: parsed["cloud_cover"] = m.group(1).replace(" ", "")

            assistant_text = parsed.get("reply") or text
            if "{" in assistant_text and "}" in assistant_text:
                try: 
                    _, clean = extract_json_from_text(assistant_text)
                    assistant_text = assistant_text.replace(clean, "").strip() 
                    if assistant_text.startswith("{") or not assistant_text: assistant_text = "I've updated the filters."
                except: pass

            return {"assistant_text": assistant_text, "parsed": parsed}
        except Exception as e:
            return {"assistant_text": f"LLM Error: {e}", "parsed": {}}

# ---------- CONTROLLER ----------
class PlanetAIAgent:
    def __init__(self, key):
        self.llm = LLMExtractor(key)
        self.geolocator = Nominatim(user_agent="planet-agent")

    def geocode(self, place):
        try:
            loc = self.geolocator.geocode(place)
            if loc: return {"lat": loc.latitude, "lon": loc.longitude}
        except: return None
        return None

    def handle_turn(self, prompt):
        # 1. Sync Uploaded Geometry to State
        if "shapefile_geometry" in st.session_state and st.session_state.shapefile_geometry:
            st.session_state.assistant_state["geometry"] = st.session_state.shapefile_geometry

        # 2. Add User Msg
        st.session_state.chat_history.append({"role":"user", "content": prompt})

        # 3. Call LLM
        res = self.llm.extract_and_reply(prompt, st.session_state.chat_history, st.session_state.assistant_state)
        
        # 4. Update State
        state = st.session_state.assistant_state
        p = res["parsed"]
        for k in ["start_date", "end_date", "cloud_cover", "place"]:
            if p.get(k): state[k] = p[k]
        
        if p.get("geometry"): state["geometry"] = parse_geometry_input(p["geometry"])

        # 5. Handle Defaulting
        if not state.get("geometry") and (p.get("decision")=="defaulted" or "assume" in prompt.lower()):
            if state.get("place"):
                loc = self.geocode(state["place"])
                if loc: state["geometry"] = create_small_bbox_polygon_from_point(loc["lat"], loc["lon"])

        st.session_state.chat_history.append({"role":"assistant", "content": res["assistant_text"]})

        # 6. Check Ready
        if p.get("decision") == "complete" or (state.get("geometry") and state.get("start_date") and state.get("end_date")):
             return {"status": "ready", "filters": state}
        return {"status": "chat"}

    def search(self, filters):
        # Build Body
        body = {"item_types":["PSScene"], "filter":{"type":"AndFilter","config":[]}}
        s = _normalize_date_iso(filters.get("start_date"), "start")
        e = _normalize_date_iso(filters.get("end_date"), "end")
        if s: body["filter"]["config"].append({"type":"DateRangeFilter","field_name":"acquired","config":{"gte":s}})
        if e: body["filter"]["config"].append({"type":"DateRangeFilter","field_name":"acquired","config":{"lte":e}})
        cc = _normalize_cloud_cover(filters.get("cloud_cover"))
        if cc is not None: body["filter"]["config"].append({"type":"RangeFilter","field_name":"cloud_cover","config":{"lte":cc}})
        if filters.get("geometry"): body["filter"]["config"].append({"type":"GeometryFilter","field_name":"geometry","config":filters["geometry"]})

        url = "https://api.planet.com/data/v1/quick-search"
        r = requests.post(url, auth=(PLANET_API_KEY, ""), json=body)
        r.raise_for_status()
        feats = r.json().get("features", [])
        save_metadata_to_db(feats)
        return feats

# ---------- VLM & Clipping ----------
def get_vlm_summary(img_bytes):
    if not HF_TOKEN: return "Error: No HF Token."
    api = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
    try:
        r = requests.post(api, headers={"Authorization": f"Bearer {HF_TOKEN}"}, data=img_bytes)
        return r.json()[0].get("generated_text", "No caption")
    except Exception as e: return f"VLM Error: {e}"

def clip_image(img_bytes, img_geom, clip_geom):
    try:
        img = Image.open(io.BytesIO(img_bytes))
        b_img = shape(img_geom).bounds
        b_clip = shape(clip_geom).bounds
        w, h = img.size
        x_scale = w / (b_img[2] - b_img[0])
        y_scale = h / (b_img[3] - b_img[1])
        left = int((b_clip[0] - b_img[0]) * x_scale)
        top = int((b_img[3] - b_clip[3]) * y_scale)
        right = int((b_clip[2] - b_img[0]) * x_scale)
        bottom = int((b_img[3] - b_clip[1]) * y_scale)
        return img.crop((max(0,left), max(0,top), min(w,right), min(h,bottom)))
    except: return Image.open(io.BytesIO(img_bytes))

def fetch_thumbnail(thumbnail_url, api_key):
    try:
        auth = HTTPBasicAuth(api_key, "")
        response = requests.get(thumbnail_url, auth=auth, timeout=30)
        response.raise_for_status()
        return response.content
    except Exception: return None

# ---------- UI ----------
def main():
    st.title("🌍 Planet API Assistant")

    if "assistant_state" not in st.session_state: st.session_state.assistant_state = {}
    if "chat_history" not in st.session_state: st.session_state.chat_history = []

    # Sidebar for Reset
    with st.sidebar:
        if st.button("🗑️ Reset Chat"):
            st.session_state.clear()
            st.rerun()

    init_db()
    agent = PlanetAIAgent(GROQ_API_KEY)

    # Main Area Upload
    with st.expander("📎 Attach Shapefile / GeoJSON (Optional)", expanded=False):
        uploaded = st.file_uploader("Upload .zip", type="zip", key="main_uploader")
        if uploaded:
            geom, err = handle_shapefile_upload(uploaded)
            if geom:
                st.session_state.shapefile_geometry = geom
                st.success("✅ Geometry Attached! The assistant will see this.")
            else:
                st.error(err)

    # Chat
    for msg in st.session_state.chat_history:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Ex: Show me images with < 10% clouds..."):
        res = agent.handle_turn(prompt)
        st.rerun()

    # Results
    if "features" in st.session_state:
        features = st.session_state.features
        st.divider()
        st.write(f"### Found {len(features)} images")
        
        c1, c2, c3, c4 = st.columns([3, 2, 2, 1])
        c1.write("**ID**"); c2.write("**Date**"); c3.write("**Cloud**"); c4.write("**Action**")
        
        for f in features[:50]:
            col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
            col1.write(f["id"])
            col2.write(f["properties"]["acquired"])
            col3.write(f["properties"]["cloud_cover"])
            
            if col4.button("👁️", key=f["id"]):
                t_url = f["_links"]["thumbnail"]
                if t_url:
                    with st.spinner("Analyzing..."):
                        raw_bytes = fetch_thumbnail(t_url, PLANET_API_KEY)
                        if raw_bytes:
                            final_image = raw_bytes
                            user_geom = st.session_state.assistant_state.get("geometry")
                            if user_geom:
                                final_image = clip_image(raw_bytes, f["geometry"], user_geom)
                            
                            summary = get_vlm_summary(final_image)
                            st.session_state.active_preview = {
                                "id": f["id"],
                                "image": final_image,
                                "summary": summary
                            }
                            st.rerun()
                        else: st.error("No image found.")

    if "active_preview" in st.session_state:
        p = st.session_state.active_preview
        with st.expander(f"Analysis: {p['id']}", expanded=True):
            st.image(p["image"], caption="Thumbnail")
            st.info(p["summary"])

    # Auto-Search
    state = st.session_state.assistant_state
    if state.get("geometry") and state.get("start_date") and "features" not in st.session_state:
        last = st.session_state.chat_history[-1] if st.session_state.chat_history else {}
        if last.get("role") == "assistant" and "decision" not in last: 
            with st.spinner("Filters complete. Searching Planet API..."):
                try:
                    feats = agent.search(state)
                    st.session_state.features = feats
                    st.rerun()
                except Exception as e:
                    st.error(f"Search Error: {e}")

if __name__ == "__main__":
    main()