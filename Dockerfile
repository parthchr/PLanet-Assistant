# Use a lightweight Python Linux image
FROM python:3.10-slim

# 1. Install system dependencies required for GeoPandas and GDAL
# (This is the magic part that fixes the "DLL load failed" errors)
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 2. Set the working directory
WORKDIR /app

# 3. Copy requirements and install Python libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of the application code
COPY . .

# 5. Expose the port Streamlit runs on
EXPOSE 8501

# 6. Command to run the app
CMD ["streamlit", "run", "planetAgent_deployed.py", "--server.port=8501", "--server.address=0.0.0.0"]