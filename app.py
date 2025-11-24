from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import datetime
import requests
from waitress import serve

app = Flask(__name__)

INTEGRATION_SERVER_URL = "http://34.227.30.110:3000"

try:
    model = YOLO('best.pt')
except:
    model = None

def fetch_all_zones():
    try:
        url = f"{INTEGRATION_SERVER_URL}/zones"
        response = requests.get(url)
        
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def fetch_plant_name(zone_id):
    try:
        url = f"{INTEGRATION_SERVER_URL}/zone/{zone_id}"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            return data.get("plant_name", "Unknown")
        return "Unknown"
    except:
        return "Unknown"

def get_plant_info_by_device(device_id):
    zones_data = fetch_all_zones()
    if not zones_data:
        return None, "Unknown"

    target_zone_id = None
    
    for zone in zones_data.get('zones', []):
        if int(device_id) in zone.get('devices', []):
            target_zone_id = zone['zone_id']
            break
    
    if target_zone_id is None:
        return None, "Unknown"

    plant_name = fetch_plant_name(target_zone_id)
    return target_zone_id, plant_name

def send_analysis_result(device_id, growth_level):
    try:
        url = f"{INTEGRATION_SERVER_URL}/analysis/growth-report"
        payload = {
            "device_id": int(device_id),
            "growth_level": int(growth_level)
        }
        response = requests.put(url, json=payload)
        return response.status_code == 200
    except:
        return False

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    device_id = request.form.get('device_id', 0)
    
    zone_id, plant_name = get_plant_info_by_device(device_id)

    try:
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        h, w, _ = img.shape
    except:
        return jsonify({'error': 'Image processing failed'}), 500

    level = 1
    
    if model:
        results = model(img, conf=0.05)
        
        if results[0].masks:
            mask = results[0].masks.data[0].cpu().numpy()
            plant_area = np.count_nonzero(mask)
            ratio = (plant_area / (mask.shape[0] * mask.shape[1])) * 100
            
            if ratio < 5: level = 1
            elif ratio < 15: level = 2
            elif ratio < 30: level = 3
            elif ratio < 50: level = 4
            elif ratio < 70: level = 5
            else: level = 6
        else:
            level = 1

    send_success = send_analysis_result(device_id, level)

    return jsonify({
        "status": "success",
        "device_id": int(device_id),
        "plant_name": plant_name,
        "growth_level": level,
        "server_sync": send_success
    })

if __name__ == '__main__':
    print("AI Server Started")
    serve(app, host='0.0.0.0', port=5000, threads=4)