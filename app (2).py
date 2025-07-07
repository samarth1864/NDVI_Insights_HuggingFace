import os
# ---------------------------------------------------
# Force-install necessary packages (if not already installed)
os.system("pip install --no-cache-dir earthengine-api")
os.system("pip install --no-cache-dir spacy")
os.system("pip install --no-cache-dir imageio")
os.system("pip install --no-cache-dir geopy")
os.system("pip install --no-cache-dir openai==0.28.0")  # Pin to 0.28.0 for the old ChatCompletion interface
os.system("pip install --no-cache-dir gradio")
os.system("pip install --no-cache-dir requests")
os.system("pip install --no-cache-dir pillow")
os.system("pip install --no-cache-dir numpy")
# ---------------------------------------------------

# --------------------- Securely Load API Keys ---------------------
# OpenAI API key from environment variable
openai_api_key = os.environ.get("OPENAI_API_KEY")
if openai_api_key:
    import openai
    openai.api_key = openai_api_key
else:
    print("Warning: OPENAI_API_KEY not found in environment variables.")

# Load GEE credentials from environment variable and write to file securely
gee_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
if gee_json:
    with open("gee_service_key.json", "w") as f:
        f.write(gee_json)
    os.chmod("gee_service_key.json", 0o600)  # restrict file permissions
else:
    print("Warning: GOOGLE_APPLICATION_CREDENTIALS_JSON not found in environment variables.")

# Load service account email securely (or use a default placeholder)
gee_email = os.environ.get("GOOGLE_SERVICE_ACCOUNT_EMAIL", "your-service-account@your-project.iam.gserviceaccount.com")

# --------------------- Now Import Other Libraries ---------------------
import ee
import json
import gradio as gr
import math
import re
from geopy.geocoders import Nominatim
import requests
from io import BytesIO
from PIL import Image
import spacy
import imageio
import numpy as np
import tempfile
import base64

# --------------------- Earth Engine Authentication ---------------------
try:
    credentials = ee.ServiceAccountCredentials(gee_email, "gee_service_key.json")
    ee.Initialize(credentials)
    print("✅ Earth Engine successfully authenticated!")
except Exception as e:
    print(f"❌ Earth Engine Initialization Failed: {e}")

# --------------------- Load spaCy Model ---------------------
spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")

# --------------------- Helper Functions ---------------------
def get_bounding_box(lat, lon, side_km=10):
    """
    Compute a square bounding box (10 km x 10 km by default) around (lat, lon).
    """
    half_side_lat = (side_km / 2) / 111
    half_side_lon = (side_km / 2) / (111 * math.cos(math.radians(lat)))
    return (lon - half_side_lon, lat - half_side_lat, lon + half_side_lon, lat + half_side_lat)

def generate_overlay(lat, lon, year):
    """
    Generate an NDVI composite image using Landsat 8 (<2022) or Landsat 9 (>=2022)
    for a 10 km x 10 km area. Returns the thumbnail URL.
    """
    min_lon, min_lat, max_lon, max_lat = get_bounding_box(lat, lon, side_km=10)
    bbox = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])
    if year < 2022:
        collection = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
                      .filterBounds(bbox)
                      .filterDate(f'{year}-01-01', f'{year}-12-31')
                      .sort('system:time_start'))
    else:
        collection = (ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
                      .filterBounds(bbox)
                      .filterDate(f'{year}-01-01', f'{year}-12-31')
                      .sort('system:time_start'))
    count = collection.size().getInfo()
    if count == 0:
        raise ValueError(f"No Landsat images found for {year} in the specified region.")
    composite = collection.median()
    ndvi = composite.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
    ndvi = ndvi.reproject(crs='EPSG:4326', scale=30)
    vis_params = {'min': 0, 'max': 1, 'palette': ['red', 'orange', 'yellow', 'green', 'blue']}
    thumb_params = {'dimensions': 333, 'region': bbox, 'format': 'png', **vis_params}
    overlay_url = ndvi.getThumbURL(thumb_params)
    print(f"Generated overlay URL for year {year}: {overlay_url}")
    return overlay_url

def generate_ndvi_difference(lat, lon, year1, year2):
    """
    Generate an NDVI difference image between two years for a 10 km x 10 km area.
    Returns the thumbnail URL.
    """
    min_lon, min_lat, max_lon, max_lat = get_bounding_box(lat, lon, side_km=10)
    bbox = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])
    def get_ndvi(year):
        if year < 2022:
            collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
        else:
            collection = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
        collection = (collection.filterBounds(bbox)
                                 .filterDate(f'{year}-01-01', f'{year}-12-31')
                                 .sort('system:time_start'))
        composite = collection.median()
        ndvi = composite.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
        return ndvi.reproject(crs='EPSG:4326', scale=30)
    ndvi1 = get_ndvi(year1)
    ndvi2 = get_ndvi(year2)
    ndvi_diff = ndvi2.subtract(ndvi1).rename('NDVI_Change')
    vis_params = {'min': -0.5, 'max': 0.5, 'palette': ['red', 'orange', 'yellow', 'green', 'blue']}
    thumb_params = {'dimensions': 333, 'region': bbox, 'format': 'png', **vis_params}
    diff_url = ndvi_diff.getThumbURL(thumb_params)
    print(f"Generated NDVI difference image for {year1} to {year2}: {diff_url}")
    return diff_url

def download_image(url):
    """
    Download an image from the URL and return it as a PIL Image.
    """
    print("Downloading image from URL:")
    print(url)
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Error downloading image from URL. Status code: {response.status_code}")
    return Image.open(BytesIO(response.content))

def image_to_base64(image):
    """
    Convert a PIL Image to a base64-encoded JPEG string.
    """
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def generate_timelapse(lat, lon, start_year, end_year):
    """
    Generate a timelapse animated GIF of NDVI maps for each year between start_year and end_year.
    The GIF is saved to a temporary file and will loop indefinitely.
    """
    frames = []
    for year in range(start_year, end_year + 1):
        try:
            url = generate_overlay(lat, lon, year)
            img = download_image(url)
            frames.append(np.array(img.convert("RGB")))
            print(f"Added frame for year {year}.")
        except Exception as e:
            print(f"Error generating frame for year {year}: {e}")
    if not frames:
        raise ValueError("No frames generated for timelapse.")
    gif_path = os.path.join(tempfile.gettempdir(), f"ndvi_timelapse_{start_year}_{end_year}.gif")
    imageio.mimsave(gif_path, frames, format='GIF', duration=1, loop=0)
    return gif_path

def parse_prompt(prompt):
    """
    Improved extraction of location and years from the prompt.
    This function:
      1. Extracts years from the prompt.
      2. Looks for a "location:" keyword.
      3. Attempts to match capitalized words after "in" or "at" (allowing up to 3 words).
      4. Uses spaCy's NER to gather geographic entities.
      5. As a final fallback, geocodes the entire prompt.
      6. Ensures proper capitalization and appends ", India" if the prompt mentions India.
    """
    years = re.findall(r"\b(1[0-9]{3}|20[0-9]{2})\b", prompt)
    if len(years) >= 2:
        historic_year = int(years[0])
        recent_year = int(years[1])
    elif len(years) == 1:
        historic_year = int(years[0])
        recent_year = 2023
    else:
        historic_year = 2013
        recent_year = 2023

    location = None
    # 1. Look for "location:" keyword.
    match = re.search(r"location:\s*([^,\.]+)", prompt, re.IGNORECASE)
    if match:
        location = match.group(1).strip()
    # 2. Try to extract capitalized words following "in" or "at" (allowing up to 3 words).
    if not location:
        match = re.search(r"\b(?:in|at)\s+(([A-Z][a-zA-Z]+(?:\s+|$)){1,3})", prompt)
        if match:
            location = match.group(1).strip()
    # 3. Use spaCy NER to detect geographic entities.
    if not location:
        doc = nlp(prompt)
        candidates = [ent.text.strip() for ent in doc.ents if ent.label_ in ["GPE", "LOC"]]
        if candidates:
            # Choose the first candidate that can be geocoded
            geolocator = Nominatim(user_agent="ndvi_app_candidate")
            for cand in candidates:
                try:
                    geo = geolocator.geocode(cand)
                    if geo:
                        location = cand
                        break
                except Exception:
                    continue
    # 4. Final fallback: try geocoding the entire prompt.
    if not location:
        geolocator = Nominatim(user_agent="ndvi_app_fallback")
        fallback = geolocator.geocode(prompt)
        if fallback:
            location = fallback.address.split(",")[0]
        else:
            location = prompt.strip()

    # Ensure proper capitalization
    location = location.title()
    # If the prompt mentions "india" but not in the location, append it.
    if "india" in prompt.lower() and "india" not in location.lower():
        location = location + ", India"
    return location, historic_year, recent_year, prompt

def generate_from_prompt(prompt):
    """
    Parse the prompt, generate NDVI maps (historic, recent, difference), a timelapse animation,
    and perform a GPT-Vision comparative analysis.
    """
    location, historic_year, recent_year, query = parse_prompt(prompt)
    if not location:
        return None, None, None, "Could not extract a location from your prompt.", None

    # Use a geolocator; if "india" is mentioned, restrict search to India.
    geolocator = Nominatim(user_agent="ndvi_app")
    if "india" in query.lower():
        loc = geolocator.geocode(location, country_codes="in")
    else:
        loc = geolocator.geocode(location)
    if not loc:
        return None, None, None, f"Location '{location}' not found.", None
    lat, lon = loc.latitude, loc.longitude
    print(f"Geocoded '{location}' to: lat {lat}, lon {lon}")

    historic_url = generate_overlay(lat, lon, historic_year)
    historic_img = download_image(historic_url)
    recent_url = generate_overlay(lat, lon, recent_year)
    recent_img = download_image(recent_url)
    diff_url = generate_ndvi_difference(lat, lon, historic_year, recent_year)
    diff_img = download_image(diff_url)

    historic_img_b64 = image_to_base64(historic_img)
    recent_img_b64 = image_to_base64(recent_img)

    analysis_prompt = (
        f"You are an expert environmental and urban planning analyst with extensive knowledge of NDVI mapping and urban trends. "
        f"Provide a detailed, in-depth report on the changes in vegetation cover in {location} between {historic_year} and {recent_year}. "
        f"Explain the colors in the output images, including the NDVI Difference Map, and identify the coordinates of the area with the most significant change. "
        f"Discuss potential causes such as urban growth, industrial development, and policy changes. User context: \"{query}\""
    )

    try:
        analysis = gpt_vision_comparison(analysis_prompt, historic_img_b64, recent_img_b64)
    except Exception as e:
        analysis = f"Error during GPT analysis: {e}"

    timelapse_gif_path = generate_timelapse(lat, lon, historic_year, recent_year)
    return historic_img, recent_img, diff_img, analysis, timelapse_gif_path

def gpt_vision_comparison(vision_prompt, image_encoded1, image_encoded2):
    """
    Use GPT-Vision to compare two NDVI maps.
    Sends a prompt and two images (as base64 strings) to OpenAI.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": vision_prompt},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + image_encoded1}},
                    {"type": "text", "text": "Below is the recent NDVI map:"},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + image_encoded2}}
                ]}
            ],
            max_tokens=800,
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error calling GPT-Vision (comparison): {e}"

# --------------------- Build the Gradio Interface ---------------------
with gr.Blocks() as demo:
    gr.Markdown("## NDVI Map Analysis with Urban Insights")
    gr.Markdown(
        "Enter a natural language prompt describing your curiosity about changes in vegetation due to urbanization. "
        "For example:<br><br>"
        "\"I'm curious about how urban growth has affected green spaces in Ahmedabad, India over the past decade.\"<br><br>"
        "The app will extract the city and relevant years (defaulting to 2013 and 2023 if not specified), "
        "generate NDVI maps (historic, recent, and a difference map), a timelapse animation, and provide an in-depth comparative analysis."
    )

    prompt_input = gr.Textbox(label="Enter your prompt", lines=4, placeholder="Type your query here...")
    analysis_btn = gr.Button("Show me the analysis")

    with gr.Row():
        historic_img_out = gr.Image(label="Historic NDVI Map", type="pil")
        recent_img_out = gr.Image(label="Recent NDVI Map", type="pil")
        diff_img_out = gr.Image(label="NDVI Difference Map", type="pil")

    analysis_out = gr.Textbox(label="GPT-Vision Comparative Analysis", lines=10)
    timelapse_out = gr.Image(label="NDVI Timelapse", type="filepath")

    analysis_btn.click(
        generate_from_prompt,
        inputs=prompt_input,
        outputs=[historic_img_out, recent_img_out, diff_img_out, analysis_out, timelapse_out]
    )

# --------------------- Launch the Gradio App ---------------------
demo.launch(server_name="0.0.0.0", server_port=7860)
