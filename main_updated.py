# planetAgent_full_fixed.py
"""
Planet API Assistant (humanlike/receptionist style)

Usage:
  - Put PLANET_API_KEY and GROQ_API_KEY in .env
  - Optionally set LLM_MODEL and LLM_TEMP in .env (defaults provided)
  - Run: streamlit run planetAgent_full_fixed.py
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
from requests.auth import HTTPBasicAuth # Added for thumbnail fetching

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

# ---------- DB ----------
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
    """Drops the metadata table and re-initializes it for a fresh start."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DROP TABLE IF EXISTS metadata")
    conn.commit()
    conn.close()
    init_db() # Recreate the empty table for the new session

def save_metadata_to_db(metadata_list):
    if not metadata_list:
        return
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
            print("DB insert error (continuing):", e)
            continue
    conn.commit()
    conn.close()

# ---------- Helpers ----------
def _normalize_date_iso(date_str, which="start"):
    if not date_str:
        return None
    s = str(date_str).strip()
    if "T" in s:
        return s if s.endswith("Z") else s + "Z"
    m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", s)
    if m:
        if which == "start":
            return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}T00:00:00.000Z"
        else:
            return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}T23:59:59.999Z"
    return s

def _normalize_cloud_cover(val):
    if val is None: return None
    try:
        v = float(val)
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
    # try JSON
    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and obj.get("type") and obj.get("coordinates"): return obj
    except Exception:
        pass
    # find bbox numbers
    m = re.search(r"\[?\s*(-?\d+(?:\.\d+)?)\s*[,\s]\s*(-?\d+(?:\.\d+)?)\s*[,\s]\s*(-?\d+(?:\.\d+)?)\s*[,\s]\s*(-?\d+(?:\.\d+)?)\s*\]?", s)
    if m:
        min_lon, min_lat, max_lon, max_lat = map(float, (m.group(1), m.group(2), m.group(3), m.group(4)))
        if min_lon > max_lon: min_lon, max_lon = max_lon, min_lon
        if min_lat > max_lat: min_lat, max_lat = max_lat, min_lat
        coords = [
            [min_lon, min_lat],
            [max_lon, min_lat],
            [max_lon, max_lat],
            [min_lon, max_lat],
            [min_lon, min_lat]
        ]
        return {"type":"Polygon","coordinates":[coords]}
    return None

def area_km2_from_bbox(min_lat, min_lon, max_lat, max_lon):
    center_lat = (min_lat + max_lat)/2.0
    height_deg = max_lat - min_lat
    width_deg = max_lon - min_lon
    height_km = abs(height_deg)*111.0
    width_km = abs(width_deg)*111.0*abs(cos(radians(center_lat)))
    return abs(width_km*height_km)

def create_small_bbox_polygon_from_point(lat, lon, half_km=2.74): # Default set to ~30km^2
    deg_lat = half_km / 111.0
    deg_lon = half_km / (111.0 * abs(cos(radians(lat)) or 1.0))
    min_lon = lon - deg_lon; max_lon = lon + deg_lon
    min_lat = lat - deg_lat; max_lat = lat + deg_lat
    coords = [[min_lon, min_lat],[max_lon, min_lat],[max_lon, max_lat],[min_lon, max_lat],[min_lon, min_lat]]
    return {"type":"Polygon","coordinates":[coords]}

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
        if not self.api_key:
            raise RuntimeError("LLM API key not configured")
        
        system_prompt = (
            "You are a warm, human receptionist-style assistant. Your job: TALK naturally and politely with the user, "
            "and collect four filters: start_date, end_date, cloud_cover, geometry (bbox or GeoJSON). "
            "If data is missing, politely ask for just that missing information (one question at a time). "
            "If the user gives a broad place (like a large city or state), you MUST ask for a more specific area, district, locality, or coordinates. Do not proceed without a more specific location unless the user explicitly tells you to 'assume' a central point. "
            "If the user explicitly says 'I don't have coordinates' or 'assume', you may set decision='defaulted' and assume a small area of about 30 km²."
            "\n\nRESPONSE REQUIREMENTS:\n"
            "- Write a short natural reply (what the user should see).\n"
            "- Include EXACTLY ONE JSON object somewhere in your reply with keys:\n"
            "  start_date, end_date, cloud_cover, geometry, place, decision, clarify, reply, reasoning\n"
            "  * reply: (string) the visible user-facing reply text (friendly)\n"
            "  * reasoning: short internal explanation (optional) — do not make reasoning the visible message.\n"
            "- decision ∈ {'complete','ask','defaulted'}.\n"
            "Keep tone friendly and concise."
        )

        messages = [{"role":"system","content":system_prompt}]
        for h in (recent_history or [])[-6:]:
            messages.append(h)
            
        messages.append({"role":"user","content":f"assistant_state = {json.dumps(assistant_state)}\n\nuser_message = {json.dumps(user_message)}"})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temp,
            "max_tokens": 600
        }

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type":"application/json"}
        r = requests.post(url, headers=headers, json=payload, timeout=40)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]

        # try to parse JSON
        parsed = None
        try:
            parsed, substr = extract_json_from_text(text)
            if isinstance(parsed, list):
                if parsed and isinstance(parsed[0], dict):
                    parsed = parsed[0] 
                else:
                    raise ValueError("LLM returned a JSON array that could not be handled.")
        except Exception:
            # fallback heuristics
            parsed = {"start_date":None,"end_date":None,"cloud_cover":None,"geometry":None,"place":None,"decision":"ask","clarify":None,"reply":None,"reasoning":None}
            dates = re.findall(r"\d{4}-\d{1,2}-\d{1,2}", text)
            if dates:
                parsed["start_date"] = dates[0]
                if len(dates) > 1: parsed["end_date"] = dates[1]
            bbox = parse_geometry_input(text) or parse_geometry_input(user_message)
            if bbox:
                parsed["geometry"] = bbox; parsed["decision"] = "complete"
            m = re.search(r"(\d{1,3}%|\d\.\d+|\d+(\.\d+)?)\s*(%|percent)?\s*cloud", text, re.I)
            if m: parsed["cloud_cover"] = m.group(1)
            pm = re.search(r"\b(?:in|at|of)\s+([A-Za-z][A-Za-z\s\-]+)", text)
            if pm: parsed["place"] = pm.group(1).strip()
            parsed["reply"] = re.sub(r"\s+"," ", text.strip())[:400]

        # prefer parsed.reply if provided; else try to compute reply from parsed.reasoning or the LLM text (clean JSON out)
        assistant_text = parsed.get("reply") or parsed.get("reasoning")
        if not assistant_text:
            try:
                _, substr = extract_json_from_text(text)
                assistant_text = text.replace(substr, "").strip()
                assistant_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", assistant_text).strip()
            except Exception:
                assistant_text = text.strip()
        if isinstance(assistant_text, str) and assistant_text.lower() in ("undefined","none","null",""):
            assistant_text = None
        if not assistant_text:
            assistant_text = "Thanks — could you provide the missing details (dates, cloud cover, or an area)?"

        # sanitize parsed keys
        for k in ["start_date","end_date","cloud_cover","geometry","place","decision","clarify","reply","reasoning"]:
            parsed.setdefault(k, None)
        if parsed.get("decision") not in ("complete","ask","defaulted"):
            parsed["decision"] = "ask"

        # ensure reply matches assistant_text if reply missing
        if not parsed.get("reply"):
            parsed["reply"] = assistant_text

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
                except Exception:
                    bbox = None
            return {"lat": loc.latitude, "lon": loc.longitude, "bbox": bbox, "display_name": raw.get("display_name")}
        except Exception:
            return None

    def search_planet_metadata(self, filters: dict):
        if not PLANET_API_KEY:
            raise RuntimeError("PLANET_API_KEY not configured")
        if isinstance(filters.get("geometry"), str):
            g = parse_geometry_input(filters["geometry"])
            if g: filters["geometry"] = g
        body = build_planet_api_body(filters)
        if not body["filter"]["config"]:
            raise ValueError("No valid filters to search.")
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
        # initialize session state
        if "assistant_state" not in st.session_state:
            st.session_state.assistant_state = {"start_date":None,"end_date":None,"cloud_cover":None,"geometry":None,"place":None}
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        # ===== FIX: Clear old results on a new user prompt =====
        if "features" in st.session_state:
            del st.session_state.features
        if "active_preview" in st.session_state:
            del st.session_state.active_preview

        # append user message to history
        st.session_state.chat_history.append({"role":"user","content":user_prompt})

        # immediate bbox detect in raw text
        geom_direct = parse_geometry_input(user_prompt)
        if geom_direct:
            st.session_state.assistant_state["geometry"] = geom_direct

        # call LLM
        try:
            out = self.llm.extract_and_reply(user_message=user_prompt, recent_history=st.session_state.chat_history, assistant_state=st.session_state.assistant_state)
        except Exception as e:
            assistant_text = f"LLM error: {e}"
            st.session_state.chat_history.append({"role":"assistant","content":assistant_text})
            return {"status":"error","assistant_text":assistant_text}

        assistant_text = out["assistant_text"]
        parsed = out["parsed"]

        # append single assistant message (LLM reply) to history
        st.session_state.chat_history.append({"role":"assistant","content":assistant_text})

        # Update assistant_state from parsed fields
        state = st.session_state.assistant_state
        if parsed.get("start_date"): state["start_date"] = parsed.get("start_date")
        if parsed.get("end_date"): state["end_date"] = parsed.get("end_date")
        if parsed.get("cloud_cover"): state["cloud_cover"] = parsed.get("cloud_cover")
        if parsed.get("place"): state["place"] = parsed.get("place")
        if parsed.get("geometry"):
            gp = parsed.get("geometry")
            gp_parsed = parse_geometry_input(gp) if isinstance(gp, str) else gp
            if gp_parsed: state["geometry"] = gp_parsed

        # If user typed that they don't have coordinates (explicit phrase), and geometry absent -> assume small area (~30 km^2)
        user_low = user_prompt.lower()
        need_assume_phrases = ["i don't have coordinates", "i dont have coordinates", "i don't know the coordinates", "i dont know the coordinates", "i don't have any coordinates", "i don't have coords", "i don't know coords", "assume", "you can assume"]
        if any(phrase in user_low for phrase in need_assume_phrases) and not state.get("geometry"):
            place = state.get("place")
            if place:
                geo = self.geocode_place(place)
                if geo and geo.get("lat") is not None:
                    lat, lon = geo["lat"], geo["lon"]
                    half_km_for_30sqkm = sqrt(30) / 2
                    small = create_small_bbox_polygon_from_point(lat, lon, half_km=half_km_for_30sqkm)
                    state["geometry"] = small
                    note = f"Okay — since you don't have coordinates, I'll assume an area of about 30 km² centered on {place}."
                    # append agent acknowledgement only if LLM didn't already say so
                    if note not in st.session_state.chat_history[-1].get("content", ""):
                        st.session_state.chat_history.append({"role":"assistant","content":note})
                        assistant_text = note

        # If LLM requested defaulted and geometry still empty but place available -> assume small area (LLM asked)
        if parsed.get("decision") == "defaulted" and not state.get("geometry"):
            place = parsed.get("place") or state.get("place")
            if place:
                geo = self.geocode_place(place)
                if geo and geo.get("lat") is not None:
                    lat, lon = geo["lat"], geo["lon"]
                    half_km_for_30sqkm = sqrt(30) / 2
                    small = create_small_bbox_polygon_from_point(lat, lon, half_km=half_km_for_30sqkm)
                    state["geometry"] = small

        # If LLM says complete -> prepare filters and return ready
        if parsed.get("decision") == "complete":
            filters = {
                "start_date": state.get("start_date"),
                "end_date": state.get("end_date"),
                "cloud_cover": state.get("cloud_cover"),
                "geometry": state.get("geometry")
            }
            return {"status":"ready","assistant_text":assistant_text,"filters":filters}

        # Else ask is needed; UI will already display LLM message
        return {"status":"need_clarify","assistant_text":assistant_text,"missing": parsed.get("clarify")}

# ---------- Build Planet quick-search body ----------
def build_planet_api_body(filters: dict):
    body = {"item_types":["PSScene"], "filter":{"type":"AndFilter","config":[]}}
    start = _normalize_date_iso(filters.get("start_date"), which="start")
    end = _normalize_date_iso(filters.get("end_date"), which="end")

    if start and end and start > end:
        start, end = end, start

    date_config = {}
    if start:
        date_config["gte"] = start
    if end:
        date_config["lte"] = end

    if date_config:
        body["filter"]["config"].append({
            "type": "DateRangeFilter",
            "field_name": "acquired",
            "config": date_config
        })

    cloud = _normalize_cloud_cover(filters.get("cloud_cover"))
    if cloud is not None:
        body["filter"]["config"].append({"type":"RangeFilter","field_name":"cloud_cover","config":{"lte":cloud}})
    
    geom = filters.get("geometry")
    if isinstance(geom, dict):
        body["filter"]["config"].append({"type":"GeometryFilter","field_name":"geometry","config":geom})
    
    return body

# ===== START: New Helper Functions for Preview & Summary =====

def fetch_thumbnail(thumbnail_url, api_key):
    """Securely fetches the thumbnail image from Planet's API."""
    try:
        auth = HTTPBasicAuth(api_key, "")
        response = requests.get(thumbnail_url, auth=auth, timeout=30)
        response.raise_for_status()
        return response.content  # Return the raw image bytes
    except Exception as e:
        print(f"Error fetching thumbnail: {e}")
        return None

def get_vlm_summary(image_bytes):
    """Sends image bytes to a local Ollama LLaVA server and gets a summary."""
    try:
        # Encode the image bytes to base64
        encoded_image = base64.b64encode(image_bytes).decode('utf-8')
        
        ollama_url = "http://localhost:11434/api/generate"
        payload = {
            "model": "llava", # Assumes 'llava' model is pulled in Ollama
            "prompt": "Describe this satellite image in a single, concise paragraph.",
            "images": [encoded_image],
            "stream": False # We want the full response at once
        }
        
        # This request might take several seconds depending on your GPU
        response = requests.post(ollama_url, json=payload, timeout=60)
        response.raise_for_status()
        
        # The response from Ollama is a JSON, summary is in the 'response' key
        summary = response.json().get("response", "No summary could be generated.")
        return summary.strip()
    
    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to the Ollama server. (Is it running?)"
    except Exception as e:
        print(f"Error getting VLM summary: {e}")
        return f"Error during VLM summary: {e}"

# ===== END: New Helper Functions =====


# ---------- Streamlit UI ----------
def main():
    st.title("🌍 Planet API Assistant (conversational)")

    # Sidebar
    st.sidebar.markdown("## Controls")
    if st.sidebar.button("Start New Chat (clear conversation & filters)"):
        # ===== FIX: Clear all session state keys on reset =====
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        reset_db()
        st.rerun()
    st.sidebar.markdown(f"LLM model: {LLM_MODEL} — temp: {LLM_TEMP}")

    init_db()
    agent = PlanetAIAgent(GROQ_API_KEY)

    # Initialize history if it doesn't exist
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display the existing chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    # Handle new user input
    if user_text := st.chat_input("Ask about satellite imagery (dates/cloud/geometry/place)..."):
        # The agent function adds the user's message to history and returns a result
        result = agent.handle_user_prompt(user_text)

        # After the agent has updated history, display the new messages manually for instant feedback
        # The user's message is at [-2], assistant's is at [-1]
        with st.chat_message("user"):
            st.markdown(st.session_state.chat_history[-2]['content'])
        
        with st.chat_message("assistant"):
            st.markdown(st.session_state.chat_history[-1]['content'])
        
        # Handle the result of the conversation (e.g., query the API)
        if result.get("status") == "ready":
            filters = result["filters"]
            
            with st.chat_message("assistant"):
                st.markdown("✅ Filters set:")
                filters_json_string = json.dumps(filters, indent=2)
                st.code(filters_json_string, language='json')
            
            with st.chat_message("assistant"):
                st.markdown("🔎 Querying Planet API with your filters now...")
            try:
                features = agent.search_planet_metadata(filters)
                count = len(features)
                st.success(f"✅ Retrieved {count} features. Saved to {DB_PATH}.")
                
                if count == 0:
                    st.info("No features returned for these filters.")
                    if "features" in st.session_state:
                        del st.session_state.features # Clear old results if any
                else:
                    # ===== FIX: Save features to session state to survive reruns =====
                    st.session_state.features = features
                
            except Exception as e:
                st.error(f"Planet API error: {e}")
                if "features" in st.session_state:
                    del st.session_state.features
        
        elif result.get("status") == "error":
            st.error(result.get("assistant_text") or "LLM/Agent error")

    # ===== FIX: Moved display logic OUTSIDE the input block =====
    # This block will now run on *every* script rerun, including button clicks.
    if "features" in st.session_state:
        features = st.session_state.features
        
        preview_list = []
        for f in features[:100]:
            properties = f.get("properties", {})
            links = f.get("_links", {})
            preview_list.append({
                "id": f.get("id"),
                "acquired": properties.get("acquired"),
                "cloud_cover": properties.get("cloud_cover"),
                "satellite": properties.get("satellite_id") or properties.get("item_type"),
                "thumbnail_url": links.get("thumbnail") # Get the thumbnail URL
            })
        
        st.markdown("---")
        st.markdown("### Results (first 100)")
        
        # Create the header row
        cols = st.columns([3, 3, 2, 2, 1])
        cols[0].markdown("**ID**")
        cols[1].markdown("**Acquired**")
        cols[2].markdown("**Cloud Cover**")
        cols[3].markdown("**Satellite**")
        cols[4].markdown("**Preview**")

        # This will hold the data for the *clicked* item
        active_preview_data = st.session_state.get("active_preview")

        # Loop through each item and create a row with a button
        for item in preview_list:
            col1, col2, col3, col4, col5 = st.columns([3, 3, 2, 2, 1])
            col1.write(item.get("id"))
            col2.write(item.get("acquired"))
            col3.write(item.get("cloud_cover"))
            col4.write(item.get("satellite"))
            
            if col5.button("👁️", key=item.get("id")):
                # This block runs when the button is clicked
                thumbnail_url = item.get("thumbnail_url")
                if thumbnail_url:
                    with st.spinner("Fetching image and generating summary... (this may take a moment)"):
                        # 1. Fetch Image
                        image_bytes = fetch_thumbnail(thumbnail_url, PLANET_API_KEY)
                        
                        if image_bytes:
                            # 2. Get VLM Summary
                            summary = get_vlm_summary(image_bytes)
                            
                            # 3. Save data to session state to display *after* rerun
                            st.session_state.active_preview = {
                                "id": item.get("id"),
                                "image_bytes": image_bytes,
                                "summary": summary
                            }
                            # Force an immediate rerun to show the expander
                            st.rerun() 
                        else:
                            st.error("Could not fetch thumbnail image from Planet API.")
                else:
                    st.error("No thumbnail URL found for this item.")
        
        # ===== FIX: Display the active preview (if one is set in state) =====
        if "active_preview" in st.session_state:
            preview_data = st.session_state.active_preview
            with st.expander(f"Preview & Summary for {preview_data['id']}", expanded=True):
                st.image(preview_data["image_bytes"], caption="Image Thumbnail")
                st.markdown("---")
                st.markdown("**VLM Summary:**")
                st.write(preview_data["summary"])

if __name__ == "__main__":
    main()