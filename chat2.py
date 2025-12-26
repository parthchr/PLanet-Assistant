import spacy
import re
import requests
from datetime import datetime, timedelta
import os
import groq
import streamlit as st

# Load the English NLP model
nlp = spacy.load("en_core_web_sm")

# --- Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "system", "content": "You are a helpful assistant that helps clarify and reason through weather or satellite-related queries. Always ask for more precise bounding boxes or dates if vague. Summarize what's missing, and suggest next steps."}
    ]

# Initialize Groq client
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = groq.Groq(api_key=GROQ_API_KEY)

# --- Parse Prompt ---
def parse_prompt(prompt):
    doc = nlp(prompt)
    bbox = None
    location = None
    pre_date = None
    post_date = None
    cloud_cover = None
    start_date = end_date = None

    try:
        dms_pattern = re.findall(r"(\d{1,2})[°º]\s?(\d{1,2})?[′']?\s?(N|S|E|W)", prompt)
        if len(dms_pattern) >= 4:
            def convert_dms(degree, minute, direction):
                decimal = float(degree) + (float(minute) / 60 if minute else 0)
                return -decimal if direction in ["S", "W"] else decimal
            coords = [
                (convert_dms(*dms_pattern[i]), convert_dms(*dms_pattern[i + 1]))
                for i in range(0, len(dms_pattern), 2)
            ]
            if len(coords) >= 2:
                (lat1, lon1), (lat2, lon2) = coords[:2]
                bbox = {
                    "south": min(lat1, lat2),
                    "north": max(lat1, lat2),
                    "west": min(lon1, lon2),
                    "east": max(lon1, lon2)
                }
    except re.error as e:
        print("Regex error in DMS pattern:", e)

    if not bbox:
        try:
            dec_match = re.findall(r"(-?\d{1,2}(?:\.\d+)?)[,\s]+(-?\d{1,3}(?:\.\d+)?)", prompt)
            if len(dec_match) >= 2:
                lat1, lon1 = map(float, dec_match[0])
                lat2, lon2 = map(float, dec_match[1])
                bbox = {
                    "south": min(lat1, lat2),
                    "north": max(lat1, lat2),
                    "west": min(lon1, lon2),
                    "east": max(lon1, lon2)
                }
        except re.error as e:
            print("Regex error in decimal pattern:", e)

    if not bbox:
        for ent in doc.ents:
            if ent.label_ in ["GPE", "LOC"]:
                location = ent.text
                break

    range_match = re.search(r"(from|between)?\s*(\w+\s+\d{1,2},?\s*\d{4})\s*(to|and|-)?\s*(\w+\s+\d{1,2},?\s*\d{4})", prompt)
    if range_match:
        try:
            start_date = datetime.strptime(range_match.group(2), "%B %d, %Y").date()
            end_date = datetime.strptime(range_match.group(4), "%B %d, %Y").date()
        except:
            pass

    month_year_match = re.search(r"(\w+)\s+(\d{4})", prompt)
    if month_year_match and not start_date:
        try:
            month, year = month_year_match.groups()
            start_date = datetime.strptime(f"{month} 1 {year}", "%B %d %Y").date()
            if "first two weeks" in prompt.lower():
                end_date = start_date + timedelta(days=13)
        except:
            pass

    cloud_match = re.search(r"(\d{1,2})\s*%.*cloud", prompt.lower())
    if cloud_match:
        cloud_cover = int(cloud_match.group(1))

    return {
        "location": location,
        "bbox": bbox,
        "pre_event_date": str(pre_date) if pre_date else None,
        "post_event_date": str(post_date) if post_date else None,
        "cloud_cover": cloud_cover,
        "start_date": str(start_date) if start_date else None,
        "end_date": str(end_date) if end_date else None
    }

# --- Reason with LLM ---
def clarify_with_llm():
    response = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=st.session_state.chat_history,
        temperature=0.5
    )
    return response.choices[0].message.content.strip()

# --- Streamlit UI ---
st.set_page_config(layout="centered")
st.title("🌦️ Natural Language Weather Query")

# Display all previous messages
for msg in st.session_state.chat_history[1:]:
    role = "assistant" if msg["role"] == "assistant" else "user"
    st.chat_message(role).write(msg["content"])

# Chat input must be evaluated after rendering previous messages
user_input = st.chat_input("Ask your weather/satellite question")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    parsed = parse_prompt(user_input)

    if not parsed['bbox']:
        with st.spinner("🤖 Reasoning through vague location..."):
            clarification = clarify_with_llm()
        st.session_state.chat_history.append({"role": "assistant", "content": clarification})
    else:
        result = f"📦 Here’s what I understood from your query:\n```json\n{parsed}\n```"
        st.session_state.chat_history.append({"role": "assistant", "content": result})
