import streamlit as st
import requests
import json
import os
from geopy.geocoders import Nominatim
from datetime import datetime, timedelta
from dotenv import load_dotenv
import streamlit_folium as stf
import folium
from shapely.geometry import shape, Point
import geojson
import geopandas as gpd
from streamlit_drawable_canvas import st_canvas
from folium import GeoJson
import time
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging
import base64
from io import BytesIO
import pandas as pd
import zipfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from pathlib import Path
from dotenv import load_dotenv

env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)
# Load environment variables
load_dotenv()
# Initialize Groq client
import groq  # Add this import to fix the NameError
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  
groq_client = groq.Groq(api_key=GROQ_API_KEY)

# Load Planet API key
PLANET_API_KEY = os.getenv("PLANET_API_KEY")

class TaskType(Enum):
    SEARCH_IMAGES = "search_images"
    DOWNLOAD_IMAGES = "download_images"
    ANALYZE_AREA = "analyze_area"
    COMPARE_DATES = "compare_dates"
    MONITOR_CHANGES = "monitor_changes"
    EXPORT_DATA = "export_data"

class Priority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class SearchCriteria:
    location: str
    date_range: Tuple[str, str]
    cloud_cover_max: Optional[float] = None
    pixel_resolution: Optional[float] = None
    sun_elevation_min: Optional[float] = None
    instrument: Optional[str] = None
    item_types: List[str] = None
    area_of_interest: Optional[Dict] = None
    max_results: int = 50

@dataclass
class Task:
    task_type: TaskType
    criteria: SearchCriteria
    priority: Priority
    description: str
    auto_download: bool = False
    asset_types: List[str] = None
    filters: Dict[str, Any] = None

class PlanetAIAgent:
    def __init__(self):
        self.session_state = st.session_state
        self.geolocator = Nominatim(user_agent="planet_ai_agent")
        
        # Initialize session state
        if "chat_history" not in self.session_state:
            self.session_state.chat_history = []
        if "task_queue" not in self.session_state:
            self.session_state.task_queue = []
        if "execution_history" not in self.session_state:
            self.session_state.execution_history = []
        if "current_context" not in self.session_state:
            self.session_state.current_context = {}
        if "search_results" not in self.session_state:
            self.session_state.search_results = []
        if "selected_image" not in self.session_state:
            self.session_state.selected_image = None
        if "current_map" not in self.session_state:
            self.session_state.current_map = None

    def _extract_location(self, prompt: str) -> str:
        """Multi-stage location extraction with fallback logic"""
        # Known location aliases and common misspellings
        location_aliases = {
            "roorki": "Roorkee",
            "roorkee": "Roorkee",
            "dehradun": "Dehradun",
            "rishikesh": "Rishikesh",
            "haridwar": "Haridwar",
            "uttarakhand": "Uttarakhand",
            "himachal": "Himachal Pradesh",
            "kashmir": "Jammu and Kashmir"
        }
        
        # Try direct matching first
        for alias, canonical in location_aliases.items():
            if alias in prompt.lower():
                return canonical
        
        # Try pattern matching
        patterns = [
              r'(?:in|of|near|at|for)\s+([A-Za-z\s,]+)(?:\s+(?:from|on|with|date|cloud)|$)',  # fixed extra closing )
              r'(?:images?|data|satellite.*?data)\s+(?:of|for|from)\s+([A-Za-z\s,]+)',
              r'(?:location.*?\s+is\s+)([A-Za-z\s,]+)'  # fixed unclosed group
            ]

        for pattern in patterns:
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                loc = match.group(1).strip()
                # Clean common false positives
                loc = re.sub(r'\b(recent|last|this|week|month|year|images?|data)\b', '', loc, flags=re.IGNORECASE).strip()
                if loc:
                    return loc
        
        # Fallback to named entity recognition via API if available
        if GROQ_API_KEY:
            try:
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                    json={
                        "model": "mistralai/mistral-7b-instruct:free",
                        "messages": [{
                            "role": "user",
                            "content": f"Extract only the location name from this text, return just the name: {prompt}"
                        }]
                    }
                )
                if response.status_code == 200:
                    loc = response.json()["choices"][0]["message"]["content"].strip()
                    if loc and loc.lower() not in ["unknown", "none", "n/a"]:
                        return loc
            except:
                pass
        
        return "Unknown"

    def _extract_date_range(self, prompt: str) -> List[str]:
        """Flexible date range extraction supporting multiple formats"""
        today = datetime.now()
        
        # Relative date patterns
        if "last week" in prompt.lower():
            return [
                (today - timedelta(days=7)).strftime("%Y-%m-%d"),
                today.strftime("%Y-%m-%d")
            ]
        elif "last month" in prompt.lower():
            return [
                (today - timedelta(days=30)).strftime("%Y-%m-%d"),
                today.strftime("%Y-%m-%d")
            ]
        
        # Absolute date patterns
        date_patterns = [
            # YYYY-MM-DD to YYYY-MM-DD
            (r'(\d{4}-\d{1,2}-\d{1,2})\s+to\s+(\d{4}-\d{1,2}-\d{1,2})', 
             lambda m: [m.group(1), m.group(2)]),
            # DD-MM-YYYY to DD-MM-YYYY
            (r'(\d{1,2})-(\d{1,2})-(\d{4})\s+to\s+(\d{1,2})-(\d{1,2})-(\d{4})',
             lambda m: [
                 f"{m.group(3)}-{m.group(2).zfill(2)}-{m.group(1).zfill(2)}",
                 f"{m.group(6)}-{m.group(5).zfill(2)}-{m.group(4).zfill(2)}"
             ]),
            # Single date variants
            (r'(\d{4}-\d{1,2}-\d{1,2})', lambda m: [m.group(1), m.group(1)]),
            (r'(\d{1,2})-(\d{1,2})-(\d{4})', 
             lambda m: [
                 f"{m.group(3)}-{m.group(2).zfill(2)}-{m.group(1).zfill(2)}",
                 f"{m.group(3)}-{m.group(2).zfill(2)}-{m.group(1).zfill(2)}"
             ])
        ]
        
        for pattern, processor in date_patterns:
            match = re.search(pattern, prompt)
            if match:
                return processor(match)
        
        # Default: last 30 days
        return [
            (today - timedelta(days=30)).strftime("%Y-%m-%d"),
            today.strftime("%Y-%m-%d")
        ]

    def _extract_cloud_cover(self, prompt: str) -> float:
        """Cloud cover extraction with percentage normalization"""
        patterns = [
            r'(?:cloud.*?cover.*?|less than|under|below)\s*(\d+)\s*%?',
            r'cloud.*?free',
            r'clear.*?sky'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                if pattern == patterns[0]:  # Percentage value
                    return min(float(match.group(1)) / 100, 1.0)
                elif pattern in patterns[1:]:  # Cloud-free terms
                    return 0.05  # Very strict threshold
        
        return 0.3  # Default 30% cloud cover

    def _classify_intent(self, prompt_lower: str) -> Tuple[str, str]:
        """Context-aware intent classification"""
        # Priority detection
        priority = "medium"
        if any(word in prompt_lower for word in ["urgent", "asap", "immediately"]):
            priority = "high"
        elif any(word in prompt_lower for word in ["emergency", "disaster", "critical"]):
            priority = "critical"
        
        # Intent classification with context awareness
        intent = "search_images"
        if any(word in prompt_lower for word in ["download", "get", "fetch"]):
            intent = "download_images"
        elif any(word in prompt_lower for word in ["analyze", "study", "examine"]):
            intent = "analyze_area"
        elif any(word in prompt_lower for word in ["compare", "difference"]):
            intent = "compare_dates"
        elif any(word in prompt_lower for word in ["monitor", "track", "observe"]):
            intent = "monitor_changes"
        elif any(word in prompt_lower for word in ["export", "csv", "geojson"]):
            intent = "export_data"
        
        return intent, priority

    def understand_prompt(self, user_prompt: str) -> Dict[str, Any]:
        """Advanced prompt understanding with multi-stage location and context extraction"""
        user_lower = user_prompt.lower()
        
        # Stage 1: Location Extraction with hierarchical matching
        location = self._extract_location(user_prompt)
        
        # Stage 2: Date Range Extraction with flexible formats
        date_range = self._extract_date_range(user_prompt)
        
        # Stage 3: Cloud Cover Extraction with percentage awareness
        cloud_cover = self._extract_cloud_cover(user_prompt)
        
        # Stage 4: Intent Classification with contextual understanding
        intent, priority = self._classify_intent(user_lower)
        
        # Special handling for disaster-related queries
        if any(word in user_lower for word in ["landslide", "earthquake", "flood", "disaster", "emergency"]):
            priority = "high"
            cloud_cover = min(cloud_cover, 0.5)  # Allow higher cloud cover for disaster monitoring
        
        return {
            "intent": intent,
            "location": location,
            "date_range": date_range,
            "cloud_cover": cloud_cover,
            "priority": priority,
            "max_results": 20,
            "item_types": ["PSScene", "Sentinel2L1C"]  # Include Sentinel-2 by default
        }

    def get_coordinates(self, location: str, buffer_km: float = 5.0) -> Optional[List[List[float]]]:
        """Enhanced geocoding with buffer zones"""
        try:
            # Try exact location first
            loc = self.geolocator.geocode(location)
            
            # If not found, try with "India" appended for Indian locations
            if not loc and location.lower() not in ["unknown", ""]:
                loc = self.geolocator.geocode(f"{location}, India")
            
            if loc:
                lat, lon = loc.latitude, loc.longitude
                
                # Convert km to degrees (approximate)
                buffer_deg = buffer_km / 111.0
                
                return [
                    [lon - buffer_deg, lat - buffer_deg],
                    [lon + buffer_deg, lat - buffer_deg],
                    [lon + buffer_deg, lat + buffer_deg],
                    [lon - buffer_deg, lat + buffer_deg],
                    [lon - buffer_deg, lat - buffer_deg]
                ]
        except Exception as e:
            logger.error(f"Geocoding error: {e}")
            st.warning(f"⚠️ Could not find location: {location}. Please try a more specific location name.")
        return None
        if not PLANET_API_KEY:
            st.error("❌ Planet API key not configured. Please set PLANET_API_KEY in your environment.")
        if not OPENWEATHER_API_KEY:
            st.error("❌ Planet API key not configured. Please set OPENWEATHER_API_KEY in your environment.")
            
        coordinates = self.get_coordinates(criteria.location)
        if not coordinates:
            st.error(f"❌ Could not find coordinates for location: {criteria.location}")
            return []

        # Build filters
        filters = []
        
        # Date filter
        if criteria.date_range[0] and criteria.date_range[1]:
            filters.append({
                "type": "DateRangeFilter",
                "field_name": "acquired",
                "config": {
                    "gte": f"{criteria.date_range[0]}T00:00:00.000Z",
                    "lte": f"{criteria.date_range[1]}T23:59:59.999Z"
                }
            })
        
        # Geometry filter
        filters.append({
            "type": "GeometryFilter",
            "field_name": "geometry",
            "config": {
                "type": "Polygon",
                "coordinates": [coordinates]
            }
        })
        
        # Cloud cover filter
        if criteria.cloud_cover_max is not None:
            filters.append({
                "type": "RangeFilter",
                "field_name": "cloud_cover",
                "config": {"lte": criteria.cloud_cover_max}
            })

        payload = {
            "item_types": criteria.item_types or ["PSScene", "Sentinel2L1C"],
            "filter": {
                "type": "AndFilter",
                "config": filters
            }
        }

        try:
            response = requests.post(
                "https://api.planet.com/data/v1/quick-search",
                auth=(PLANET_API_KEY , ""),
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=30
            )

            if response.status_code == 200:
                features = response.json().get("features", [])
                # Sort by acquisition date (newest first)
                features.sort(key=lambda x: x.get("properties", {}).get("acquired", ""), reverse=True)
                return features[:criteria.max_results]
            else:
                error_msg = f"Planet API Error {response.status_code}"
                try:
                    error_detail = response.json().get("message", response.text)
                    error_msg += f": {error_detail}"
                except Exception:
                    error_msg += f": {response.text}"
                st.error(f"🌍 {error_msg}")
                return []
        except requests.exceptions.Timeout:
            st.error("⏱️ Search request timed out. Please try again.")
            return []
        except Exception as e:
            logger.error(f"Planet API search error: {e}")
            st.error(f"❌ Search failed: {str(e)}")
            return []

    def display_search_results(self, images: List[Dict], location: str) -> None:
        """Display search results in an interactive format"""
        if not images:
            st.warning("🔍 No images found matching your criteria. Try adjusting the location, date range, or cloud cover threshold.")
            return
        
        st.success(f"📡 Found {len(images)} satellite images for **{location}**")
        
        # Store results in session state
        self.session_state.search_results = images
        
        # Create interactive results display
        st.subheader("🖼️ Search Results")
        
        # Create columns for layout
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("**📋 Available Images**")
            
            # Create a container for the image list
            for i, img in enumerate(images):
                props = img.get("properties", {})
                links = img.get("_links", {})
                
                # Format date
                acquired_date = props.get("acquired", "Unknown")
                if acquired_date != "Unknown":
                    try:
                        date_obj = datetime.fromisoformat(acquired_date.replace('Z', '+00:00'))
                        formatted_date = date_obj.strftime("%Y-%m-%d %H:%M")
                    except:
                        formatted_date = acquired_date[:16]
                else:
                    formatted_date = "Unknown"
                
                # Create clickable image card
                with st.container():
                    # Show thumbnail if available
                    if links.get("thumbnail"):
                        try:
                            thumb_response = requests.get(links["thumbnail"], timeout=10)
                            if thumb_response.status_code == 200:
                                st.image(thumb_response.content, width=150)
                        except:
                            st.image("https://via.placeholder.com/150x150/cccccc/666666?text=No+Image", width=150)
                    
                    # Image info
                    st.markdown(f"""
                    **Image {i+1}**  
                    📅 **Date:** {formatted_date}  
                    ☁️ **Cloud:** {round(props.get('cloud_cover', 0) * 100, 1)}%  
                    🌞 **Sun Elev:** {round(props.get('sun_elevation', 0), 1)}°  
                    🛰️ **Satellite:** {props.get('satellite_id', 'Unknown')}  
                    📏 **Resolution:** {props.get('pixel_resolution', 'N/A')} m
                    """)
                    
                    # Action buttons
                    col_view, col_down = st.columns(2)
                    with col_view:
                        if st.button(f"🗺️ View", key=f"view_{i}", use_container_width=True):
                            self.session_state.selected_image = i
                            st.rerun()
                    
                    with col_down:
                        if st.button(f"⬇️ Download", key=f"download_{i}", use_container_width=True):
                            self.download_single_image(img, i)
                    
                    st.divider()
        
        with col2:
            # Map display
            st.markdown("**🗺️ Image Footprints & Location**")
            
            # Create base map
            coordinates = self.get_coordinates(location)
            if coordinates:
                center_lat = sum(coord[1] for coord in coordinates) / len(coordinates)
                center_lon = sum(coord[0] for coord in coordinates) / len(coordinates)
                
                m = folium.Map(
                    location=[center_lat, center_lon],
                    zoom_start=10,
                    tiles='OpenStreetMap'
                )
                
                # Add search area
                folium.Polygon(
                    locations=[(coord[1], coord[0]) for coord in coordinates],
                    color="red",
                    fill=False,
                    weight=3,
                    popup=f"Search Area: {location}"
                ).add_to(m)
                
                # Add image footprints
                colors = ['blue', 'green', 'purple', 'orange', 'red', 'pink', 'gray', 'darkblue', 'darkgreen', 'cadetblue']
                
                for i, img in enumerate(images):
                    geom = img.get("geometry", {})
                    props = img.get("properties", {})
                    
                    if geom and geom.get("coordinates"):
                        # Get coordinates and handle different geometry types
                        coords = geom["coordinates"]
                        if geom["type"] == "Polygon":
                            locations = [(y, x) for x, y in coords[0]]
                        else:
                            continue
                        
                        color = colors[i % len(colors)]
                        
                        # Highlight selected image
                        if hasattr(self.session_state, 'selected_image') and self.session_state.selected_image == i:
                            color = 'yellow'
                            weight = 4
                            opacity = 0.8
                        else:
                            weight = 2
                            opacity = 0.5
                        
                        # Add polygon
                        folium.Polygon(
                            locations=locations,
                            color=color,
                            fill=True,
                            fillOpacity=0.2,
                            weight=weight,
                            opacity=opacity,
                            popup=f"""
                            <b>Image {i+1}</b><br>
                            Date: {props.get('acquired', 'Unknown')[:10]}<br>
                            Cloud Cover: {round(props.get('cloud_cover', 0) * 100, 1)}%<br>
                            Satellite: {props.get('satellite_id', 'Unknown')}
                            """
                        ).add_to(m)
                
                # Display the map
                map_data = stf.folium_static(m, width=600, height=400)
                
                # Selected image details
                if hasattr(self.session_state, 'selected_image') and self.session_state.selected_image is not None:
                    selected_idx = self.session_state.selected_image
                    if 0 <= selected_idx < len(images):
                        selected_img = images[selected_idx]
                        self.display_selected_image_details(selected_img, selected_idx + 1)
        
        # Bulk actions
        st.subheader("🔧 Bulk Actions")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 Download All Images", use_container_width=True):
                self.bulk_download_images(images)
        
        with col2:
            if st.button("📊 Export Metadata", use_container_width=True):
                self.export_metadata(images, location)
        
        with col3:
            if st.button("🗺️ Export Footprints", use_container_width=True):
                self.export_footprints(images, location)

    def display_selected_image_details(self, image: Dict, image_num: int) -> None:
        """Display detailed information about the selected image"""
        st.markdown("---")
        st.subheader(f"🔍 Selected Image {image_num} Details")
        
        props = image.get("properties", {})
        links = image.get("_links", {})
        
        # Create detailed info display
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📋 Basic Information**")
            st.write(f"**Acquisition Date:** {props.get('acquired', 'Unknown')}")
            st.write(f"**Cloud Cover:** {round(props.get('cloud_cover', 0) * 100, 2)}%")
            st.write(f"**Sun Elevation:** {round(props.get('sun_elevation', 0), 2)}°")
            st.write(f"**Sun Azimuth:** {round(props.get('sun_azimuth', 0), 2)}°")
            st.write(f"**Satellite ID:** {props.get('satellite_id', 'Unknown')}")
            st.write(f"**Instrument:** {props.get('instrument', 'Unknown')}")
        
        with col2:
            st.markdown("**🎯 Technical Details**")
            st.write(f"**Pixel Resolution:** {props.get('pixel_resolution', 'Unknown')} m")
            st.write(f"**Quality Category:** {props.get('quality_category', 'Unknown')}")
            st.write(f"**Strip ID:** {props.get('strip_id', 'Unknown')}")
            st.write(f"**View Angle:** {round(props.get('view_angle', 0), 2)}°")
            st.write(f"**Ground Control:** {props.get('ground_control', 'Unknown')}")
        
        # Show larger thumbnail
        if links.get("thumbnail"):
            st.markdown("**🖼️ Preview**")
            try:
                thumb_response = requests.get(links["thumbnail"], timeout=10)
                if thumb_response.status_code == 200:
                    st.image(thumb_response.content, width=400, caption=f"Image {image_num} Preview")
            except:
                st.warning("Could not load image preview")
        
        # Download options
        st.markdown("**⬇️ Download Options**")
        asset_types = ["visual", "analytic", "analytic_sr", "udm", "udm2"]
        selected_asset = st.selectbox("Select Asset Type:", asset_types, key=f"asset_select_{image_num}")
        
        if st.button(f"Download {selected_asset.title()} Asset", key=f"download_asset_{image_num}"):
            self.download_single_image(image, image_num - 1, selected_asset)

    def search_planet_images(self, criteria):
         if not PLANET_API_KEY:
             st.error("❌ Planet API key not configured. Please set PLANET_API_KEY in your environment.")
             return []

         coordinates = self.get_coordinates(criteria.location)
         if not coordinates:
              st.error(f"❌ Could not find coordinates for location: {criteria.location}")
              return []

    # Build filters
         filters = []

    # Date filter
         start_date, end_date = criteria.date_range
         if start_date and end_date:
              filters.append({
                  "type": "DateRangeFilter",
                  "field_name": "acquired",
                   "config": {
                          "gte": f"{start_date}T00:00:00.000Z",
                          "lte": f"{end_date}T23:59:59.999Z"
                        }
                   }
                             )

    # Geometry filter
         filters.append({
              "type": "GeometryFilter",
              "field_name": "geometry",
              "config": {
                 "type": "Polygon",
                 "coordinates": [coordinates]
                } 
              }
                        )

    # Cloud cover filter
         if criteria.cloud_cover_max is not None:
             filters.append({
                 "type": "RangeFilter",
                 "field_name": "cloud_cover",
                 "config": {"lte": criteria.cloud_cover_max}
                 }
                            )

         payload = {
             "item_types": criteria.item_types or ["PSScene", "Sentinel2L1C"],
             "filter": {
                 "type": "AndFilter",
                 "config": filters
                }
         }

         try:
             response = requests.post(
                 "https://api.planet.com/data/v1/quick-search",
                 auth=(PLANET_API_KEY, ""),
                 headers={"Content-Type": "application/json"},
                 data=json.dumps(payload),
                 timeout=30
             )

             if response.status_code == 200:
                 features = response.json().get("features", [])
                 features.sort(key=lambda x: x.get("properties", {}).get("acquired", ""), reverse=True)
                 return features[:criteria.max_results]
             else:
                 error_msg = f"Planet API Error {response.status_code}: {response.text}"
                 st.error(error_msg)
                 return []
         except Exception as e:
             logger.error(f"Planet API search error: {e}")
             st.error(f"❌ Search failed: {str(e)}")
             return []



    def download_single_image(self, image: Dict, index: int, asset_type: str = "visual") -> None:
        """Fixed download function with complete session state preservation"""
        # Store current state before any download operations
        current_state = {
            'search_results': self.session_state.get('search_results', []),
            'selected_image': self.session_state.get('selected_image', None),
            'chat_history': self.session_state.get('chat_history', [])
        }

        try:
            props = image.get("properties", {})
            links = image.get("_links", {})
            
            if not links.get("assets"):
                st.error("❌ No assets available for this image")
                return

            with st.spinner(f"🔄 Preparing download for Image {index + 1}..."):
                # Get asset information
                asset_response = requests.get(
                    links["assets"],
                    timeout=30
                )
                
                if asset_response.status_code != 200:
                    st.error(f"❌ Failed to get asset information: {asset_response.status_code}")
                    return
                
                assets = asset_response.json()
                
                if asset_type not in assets:
                    available_assets = list(assets.keys())
                    st.error(f"❌ Asset type '{asset_type}' not available. Available: {available_assets}")
                    return
                
                asset_info = assets[asset_type]
                
                # Asset activation process
                if asset_info.get("status") != "active":
                    st.info("⏳ Asset not active. Activating...")
                    activate_url = asset_info.get("_links", {}).get("activate")
                    
                    if not activate_url:
                        st.error("❌ No activation URL available for this asset.")
                        return
                    activate_response = requests.post(
                        activate_url,
                        auth=(PLANET_API_KEY , ""),
                        timeout=30
                    )
                    
                    if activate_response.status_code not in [200, 202]:
                        st.error(f"❌ Failed to activate asset: {activate_response.status_code}")
                        return
                    
                    # Activation progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for attempt in range(30):  # 5 minute timeout
                        status_text.text(f"Activating asset... {attempt + 1}/30")
                        status_response = requests.get(
                            links["assets"],
                            auth=(PLANET_API_KEY , ""),
                            timeout=30
                        )
                        if status_response.status_code == 200:
                            updated_assets = status_response.json()
                            if updated_assets[asset_type].get("status") == "active":
                                asset_info = updated_assets[asset_type]
                                break
                    else:
                        progress_bar.empty()
                        status_text.empty()
                        st.error("⏱️ Asset activation timed out. Please try again later.")
                        return
                    
                    progress_bar.empty()
                    status_text.empty()
                    st.success("✅ Asset activated successfully!")

                # Download process
                download_url = asset_info.get("location")
                if not download_url:
                    st.error("❌ No download URL available")
                    return
                
                with st.spinner("⬇️ Downloading image..."):
                    download_response = requests.get(download_url, stream=True, timeout=300)
                    
                    if download_response.status_code != 200:
                        st.error(f"❌ Download failed: {download_response.status_code}")
                        return
                    
                    # Generate filename
                    acquired_date = props.get("acquired", "unknown")[:10]
                    satellite_id = props.get("satellite_id", "unknown")
                    filename = f"{satellite_id}_{acquired_date}_{asset_type}.tif"
                    
                    # Download with progress tracking
                    total_size = int(download_response.headers.get('content-length', 0))
                    if total_size > 0:
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        with open(filename, "wb") as f:
                            downloaded = 0
                            for chunk in download_response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                                    downloaded += len(chunk)
                                    progress = downloaded / total_size
                                    progress_bar.progress(progress)
                                    status_text.text(
                                        f"Downloaded: {downloaded / 1024 / 1024:.1f} MB / "
                                        f"{total_size / 1024 / 1024:.1f} MB"
                                    )
                        
                        progress_bar.empty()
                        status_text.empty()
                    else:
                        with open(filename, "wb") as f:
                            for chunk in download_response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                    
                    file_size = os.path.getsize(filename)
                    st.success(f"✅ Downloaded: {filename} ({file_size / 1024 / 1024:.1f} MB)")
                    
                    # Create download button
                    with open(filename, "rb") as f:
                        st.download_button(
                            label=f"📥 Save {filename}",
                            data=f.read(),
                            file_name=filename,
                            mime="image/tiff",
                            key=f"download_{index}_{int(time.time())}"  # Unique key to prevent caching
                        )
        except requests.exceptions.Timeout:
            st.error("⏱️ Download request timed out. Please try again.")
        except Exception as e:
            logger.error(f"Download error: {str(e)}")
            st.error(f"❌ Download failed: {str(e)}")
        finally:
            # Restore state in all cases
            self.session_state.search_results = current_state['search_results']
            self.session_state.selected_image = current_state['selected_image']
            self.session_state.chat_history = current_state['chat_history']
            st.rerun()  # Refresh UI while preserving state

    def bulk_download_images(self, images: List[Dict]) -> None:
        """Download multiple images with progress tracking"""
        st.info(f"🚀 Starting bulk download of {len(images)} images...")
        
        success_count = 0
        failed_count = 0
        
        # Create progress containers
        overall_progress = st.progress(0)
        status_container = st.empty()
        
        for i, image in enumerate(images):
            try:
                status_container.text(f"Processing image {i+1}/{len(images)}")
                
                props = image.get("properties", {})
                acquired_date = props.get("acquired", "unknown")[:10]
                satellite_id = props.get("satellite_id", "unknown")
                
                # Download with default visual asset
                self.download_single_image(image, i, "visual")
                success_count += 1
    
            except Exception as e:
                logger.error(f"Failed to download image {i+1}: {e}")
                failed_count += 1
                
            # Update progress
            overall_progress.progress((i + 1) / len(images))
        
        # Final status
        overall_progress.empty()
        status_container.empty()
        
        if success_count > 0:
            st.success(f"✅ Successfully downloaded {success_count} images")
        if failed_count > 0:
            st.error(f"❌ Failed to download {failed_count} images")

    def export_metadata(self, images: List[Dict], location: str) -> None:
        """Export metadata to CSV"""
        try:
            metadata_list = []
            
            for i, img in enumerate(images):
                props = img.get("properties", {})
                geom = img.get("geometry", {})
                
                metadata = {
                    "image_id": props.get("id", f"image_{i}"),
                    "location": location,
                    "acquired_date": props.get("acquired", ""),
                    "cloud_cover": props.get("cloud_cover", 0),
                    "sun_elevation": props.get("sun_elevation", 0),
                    "sun_azimuth": props.get("sun_azimuth", 0),
                    "satellite_id": props.get("satellite_id", ""),
                    "instrument": props.get("instrument", ""),
                    "pixel_resolution": props.get("pixel_resolution", 0),
                    "quality_category": props.get("quality_category", ""),
                    "view_angle": props.get("view_angle", 0),
                    "geometry_type": geom.get("type", ""),
                    "coordinates": str(geom.get("coordinates", []))
                }
                metadata_list.append(metadata)
            
            # Create DataFrame
            df = pd.DataFrame(metadata_list)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"planet_metadata_{location.replace(' ', '_')}_{timestamp}.csv"
            
            # Convert to CSV
            csv_data = df.to_csv(index=False)
            
            # Provide download
            st.download_button(
                label="📊 Download Metadata CSV",
                data=csv_data,
                file_name=filename,
                mime="text/csv"
            )
            
            st.success(f"✅ Metadata exported: {len(metadata_list)} records")
            
        except Exception as e:
            logger.error(f"Metadata export error: {e}")
            st.error(f"❌ Failed to export metadata: {str(e)}")

    def export_footprints(self, images: List[Dict], location: str) -> None:
        """Export image footprints as GeoJSON"""
        try:
            features = []
            
            for i, img in enumerate(images):
                props = img.get("properties", {})
                geom = img.get("geometry", {})
                
                if geom:
                    feature = {
                        "type": "Feature",
                        "geometry": geom,
                        "properties": {
                            "image_id": props.get("id", f"image_{i}"),
                            "acquired_date": props.get("acquired", ""),
                            "cloud_cover": props.get("cloud_cover", 0),
                            "satellite_id": props.get("satellite_id", ""),
                            "pixel_resolution": props.get("pixel_resolution", 0)
                        }
                    }
                    features.append(feature)
            
            # Create GeoJSON
            geojson_data = {
                "type": "FeatureCollection",
                "features": features
            }
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"planet_footprints_{location.replace(' ', '_')}_{timestamp}.geojson"
            
            # Convert to JSON string
            json_data = json.dumps(geojson_data, indent=2)
            
            # Provide download
            st.download_button(
                label="🗺️ Download Footprints GeoJSON",
                data=json_data,
                file_name=filename,
                mime="application/json"
            )
            
            st.success(f"✅ Footprints exported: {len(features)} features")
            
        except Exception as e:
            logger.error(f"Footprints export error: {e}")
            st.error(f"❌ Failed to export footprints: {str(e)}")

    def generate_ai_response(self, user_message: str, search_results: List[Dict] = None) -> str:
        """Generate AI response using OpenRouter API"""
        if not GROQ_API_KEY:
            return "🤖 AI response generation requires OpenRouter API key configuration."
        
        try:
            # Prepare context
            context = f"User query: {user_message}\n"
            
            if search_results:
                context += f"\nFound {len(search_results)} satellite images.\n"
                context += "Image summary:\n"
                
                for i, img in enumerate(search_results[:5]):  # Limit to first 5
                    props = img.get("properties", {})
                    context += f"- Image {i+1}: {props.get('acquired', 'Unknown')[:10]}, "
                    context += f"Cloud: {round(props.get('cloud_cover', 0) * 100, 1)}%, "
                    context += f"Satellite: {props.get('satellite_id', 'Unknown')}\n"
            
            # Prepare prompt
            system_prompt = """
                             You are a helpful assistant that helps clarify vague satellite image queries.
                             Always ask for more precise locations or dates if the user is vague.
                             Also ask the user for the bounding box coordinates in the end after reasoning with the user.
                             Also ask the post and pre dates if the user is vague about the dates.
                             Avoid assuming — always verify vague queries.
                             """
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context}
            ]
            
            # Call OpenRouter API
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama3-70b-8192",
                    "messages": messages,
                    "max_tokens": 500
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                return f"🤖 AI response error: {response.status_code}"
                
        except Exception as e:
            logger.error(f"AI response error: {e}")
            return "🤖 AI response generation failed. Please try again."

    def run_chat_interface(self):
        """Main chat interface"""
        st.title("🛰️ Planet AI Agent")
        st.markdown("Ask me to search for satellite images, analyze locations, or download data!")
        
        # Display chat history
        for message in self.session_state.chat_history:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(message["content"])
        
        # Chat input
        if prompt := st.chat_input("What satellite data are you looking for?"):
            # Add user message to chat history
            self.session_state.chat_history.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.write(prompt)
            
            # Process the request
            with st.chat_message("assistant"):  
                with st.spinner("🔄 Processing your request..."):
                    # Understand the prompt
                    intent_data = self.understand_prompt(prompt)
                    
                    # Create search criteria
                    criteria = SearchCriteria(
                        location=intent_data["location"],
                        date_range=(intent_data["date_range"][0], intent_data["date_range"][1]),
                        cloud_cover_max=intent_data["cloud_cover"],
                        max_results=intent_data["max_results"],
                        item_types=intent_data["item_types"]
                    )
                    
                    # Search for images
                    search_results = self.search_planet_images(criteria)
                    
                    # Generate AI response
                    ai_response = self.generate_ai_response(prompt, search_results)
                    st.write(ai_response)
                    
                    # Display results if found
                    if search_results:
                        self.display_search_results(search_results, intent_data["location"])
                    
                    # Add assistant response to chat history
                    self.session_state.chat_history.append({"role": "assistant", "content": ai_response})

    def run_sidebar_controls(self):
        """Sidebar controls and settings"""
        with st.sidebar:
            st.header("🛠️ Controls")
            if PLANET_API_KEY:
                st.success("✅ Planet API Connected")
            else:
                st.error("❌ Planet API Key Missing")
                
            if GROQ_API_KEY:
                st.success("✅ OpenRouter API Connected")
            else:
                st.warning("⚠️ OpenRouter API Key Missing")
            
            st.divider()
            
                      
            # Settings
            st.subheader("⚙️ Settings")
            
            if st.button("🗑️ Clear Chat History", use_container_width=True):
                self.session_state.chat_history = []
                st.rerun()
            
            if st.button("🔄 Reset Session", use_container_width=True):
                for key in list(self.session_state.keys()):
                    del self.session_state[key]
                st.rerun()
            
            # Statistics
            st.subheader("📊 Session Stats")
            st.metric("Chat Messages", len(self.session_state.chat_history))
            if hasattr(self.session_state, 'search_results'):
                st.metric("Images Found", len(self.session_state.search_results))

def main():
    """Main application entry point"""
    st.set_page_config(
        page_title="Planet AI Agent",
        page_icon="🛰️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize the agent
    agent = PlanetAIAgent()
    
    # Run sidebar controls
    agent.run_sidebar_controls()
    
    # Run main chat interface
    agent.run_chat_interface()
    
    # Footer
    st.markdown("---")
    st.markdown("🛰️ **Planet AI Agent** - Powered by Planet Labs API & OpenRouter AI")

if __name__ == "__main__":
    main()