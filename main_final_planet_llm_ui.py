# Final script with geocoding support for location using Nominatim (OpenStreetMap)
import streamlit as st
import os
import requests
import re
from datetime import datetime, timedelta
import spacy
import groq
from dotenv import load_dotenv

load_dotenv()
nlp = spacy.load("en_core_web_sm")
groq_client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))

# 🌍 Geocode place name into lat/lon using OpenStreetMap Nominatim API
def geocode_location(name):
    try:
        url = f"https://nominatim.openstreetmap.org/search?q={name}&format=json&limit=1"
        headers = {"User-Agent": "planet-weather-agent"}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if len(data) > 0:
                lat = float(data[0]["lat"])
                lon = float(data[0]["lon"])
                # Return bounding box (0.25 deg buffer)
                return {
                    "south": lat - 0.25,
                    "north": lat + 0.25,
                    "west": lon - 0.25,
                    "east": lon + 0.25
                }
    except:
        return None
    return None

def clarify_with_llm():
    response = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=st.session_state.chat_history,
        temperature=0.5
    )
    return response.choices[0].message.content.strip()

def parse_prompt(prompt):
    doc = nlp(prompt)
    bbox = None
    location = None
    pre_date = None
    post_date = None
    cloud_cover = None
    start_date = end_date = None

    def dms_to_decimal(dms_str):
        match = re.match(r"(\\d+)[°:](\\d+)[′']?([NSWE])", dms_str)
        if not match:
            return None
        degrees = int(match.group(1))
        minutes = int(match.group(2))
        direction = match.group(3).upper()
        decimal = degrees + minutes / 60
        if direction in ['S', 'W']:
            decimal *= -1
        return decimal

    dms_coords = re.findall(r"(\\d{1,2})[°:](\\d{1,2})['′]?[NSWE]", prompt)
    if len(dms_coords) >= 4:
        coord_strs = re.findall(r"\\d{1,2}[°:]\\d{1,2}['′]?[NSWE]", prompt)
        lat1 = dms_to_decimal(coord_strs[0])
        lat2 = dms_to_decimal(coord_strs[1])
        lon1 = dms_to_decimal(coord_strs[2])
        lon2 = dms_to_decimal(coord_strs[3])
        if all(v is not None for v in [lat1, lat2, lon1, lon2]):
            bbox = {
                "south": min(lat1, lat2),
                "north": max(lat1, lat2),
                "west": min(lon1, lon2),
                "east": max(lon1, lon2)
            }

    if not bbox:
        bbox_match = re.findall(r"(?<!\\d)(-?\\d{1,2}(?:\\.\\d+)?)\\s*,\\s*(-?\\d{1,3}(?:\\.\\d+)?)(?!\\d)", prompt)
        if len(bbox_match) >= 2:
            lat1, lon1 = map(float, bbox_match[0])
            lat2, lon2 = map(float, bbox_match[1])
            bbox = {
                "south": min(lat1, lat2),
                "north": max(lat1, lat2),
                "west": min(lon1, lon2),
                "east": max(lon1, lon2)
            }

    for ent in doc.ents:
        if ent.label_ in ["GPE", "LOC"]:
            location = ent.text
            break

    range_match = re.search(r"(from|between)?\\s*(\\w+\\s+\\d{1,2},?\\s*\\d{4})\\s*(to|and|-)?\\s*(\\w+\\s+\\d{1,2},?\\s*\\d{4})", prompt, re.IGNORECASE)
    if range_match:
        try:
            start_date = datetime.strptime(range_match.group(2), "%B %d, %Y").date()
            end_date = datetime.strptime(range_match.group(4), "%B %d, %Y").date()
        except ValueError:
            pass

    month_range = re.search(r"(between|from)?\\s*(January|February|March|April|May|June|July|August|September|October|November|December)\\s+(\\d{4})\\s*(to|and|-)\\s*(January|February|March|April|May|June|July|August|September|October|November|December)\\s+(\\d{4})", prompt, re.IGNORECASE)
    if month_range:
        try:
            start_date = datetime.strptime(f"{month_range.group(2)} {month_range.group(3)}", "%B %Y").date()
            end_date = datetime.strptime(f"{month_range.group(5)} {month_range.group(6)}", "%B %Y").date()
        except ValueError:
            pass

    date_matches = re.findall(r"(before|after)?\\s*(January|February|March|April|May|June|July|August|September|October|November|December)\\s+(\\d{4})", prompt, re.IGNORECASE)
    for modifier, month, year in date_matches:
        dt = datetime.strptime(f"{month} {year}", "%B %Y").date()
        if modifier and modifier.lower() == "before":
            pre_date = dt
        elif modifier and modifier.lower() == "after":
            post_date = dt

    cloud_match = re.search(r"(\\d{1,2})\\s*%.*cloud", prompt.lower())
    if cloud_match:
        cloud_cover = int(cloud_match.group(1))

    return {
        "location": location,
        "bbox": bbox,
        "pre_event_date": str(pre_date) if pre_date else None,
        "post_event_date": str(post_date) if post_date else None,
        "cloud_cover": cloud_cover,
        "start_date": str(start_date) if start_date else None,
        "end_date": str(end_date) if end_date else None,
    }

def call_planet_api(parsed, api_key):
    bbox = parsed.get('bbox')
    start = parsed.get("start_date")
    end = parsed.get("end_date")
    cloud_cover = parsed.get("cloud_cover")

    if not bbox or not start or not end:
        return {"error": "Missing bbox or date range to search Planet data."}

    geometry = {
        "type": "Polygon",
        "coordinates": [[
            [bbox["west"], bbox["south"]],
            [bbox["east"], bbox["south"]],
            [bbox["east"], bbox["north"]],
            [bbox["west"], bbox["north"]],
            [bbox["west"], bbox["south"]]
        ]]
    }

    filters = {
        "type": "AndFilter",
        "config": [
            {"type": "GeometryFilter", "field_name": "geometry", "config": geometry},
            {"type": "DateRangeFilter", "field_name": "acquired", "config": {
                "gte": f"{start}T00:00:00.000Z",
                "lte": f"{end}T23:59:59.999Z"
            }},
            {"type": "RangeFilter", "field_name": "cloud_cover", "config": {
                "lte": cloud_cover / 100 if cloud_cover else 0.3
            }}
        ]
    }

    search_request = {"item_types": ["PSScene"], "filter": filters}
    response = requests.post("https://api.planet.com/data/v1/quick-search", auth=(api_key, ""), json=search_request)

    if response.status_code != 200:
        return {"error": f"Planet API error: {response.status_code}", "details": response.text}

    results = response.json().get("features", [])[:10]
    return [ {
        "id": f["id"],
        "acquired": f["properties"]["acquired"],
        "cloud_cover": f["properties"].get("cloud_cover"),
        "thumbnail": f["_links"].get("thumbnail"),
        "item_type": f["properties"].get("item_type")
    } for f in results ]

# --- Streamlit App ---
st.title("🛰️ LLM + Geospatial Image Assistant")
API_KEY = os.getenv("PLANET_API_KEY")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{"role": "system", "content": "You are helping users search for satellite images using natural language. Ask for bounding box or location."}]

user_input = st.text_input("📝 What kind of satellite data are you looking for?")
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    parsed = parse_prompt(user_input)

    # Geocode fallback if bbox is missing but location is available
    if not parsed["bbox"] and parsed["location"]:
        geocoded_bbox = geocode_location(parsed["location"])
        if geocoded_bbox:
            parsed["bbox"] = geocoded_bbox
            st.success(f"📍 Location '{parsed['location']}' converted to bounding box using OpenStreetMap.")
        else:
            st.warning("⚠️ Failed to geocode the location.")

    # If still incomplete, ask LLM to clarify
    if not parsed["bbox"] or not parsed["start_date"] or not parsed["end_date"]:
        with st.spinner("🤖 LLM is clarifying your vague input..."):
            clarification = clarify_with_llm()
        st.session_state.chat_history.append({"role": "assistant", "content": clarification})
        st.info(clarification)
    else:
        st.success("✅ Your input is ready for Planet API search.")
        if st.button("📡 Fetch Satellite Images"):
            results = call_planet_api(parsed, API_KEY)
            if isinstance(results, list):
                for r in results:
                    st.markdown(f"**ID:** {r['id']}")
                    st.markdown(f"**Acquired:** {r['acquired']}")
                    st.markdown(f"**Cloud Cover:** {r['cloud_cover'] * 100:.2f}%")
                    if r["thumbnail"]:
                        st.image(r["thumbnail"], width=400)
                    st.markdown("---")
            else:
                st.error(results)

