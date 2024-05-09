import streamlit as st
from azure.storage.blob import BlobServiceClient
from PIL import Image
import io

# Initialize connection to Azure Blob Storage
connect_str = "DefaultEndpointsProtocol=https;AccountName=515team2;AccountKey=+wc53G0GKd551uGI/gn+ow5YcrqralBanMwl+MqJoxReUPwSHwBE6wu4Eoh3awBwxR4za3qlC0hQ+AStlJ2PmA==;EndpointSuffix=core.windows.net"  # Replace with your actual connection string
blob_service_client = BlobServiceClient.from_connection_string(connect_str)
container_client = blob_service_client.get_container_client("imagetest")
sensor_data_client = blob_service_client.get_container_client("bmetest")

def load_image_from_blob(blob_client):
    img_data = blob_client.download_blob().readall()
    return Image.open(io.BytesIO(img_data))

def load_sensor_data(sensor_data_client):
    sensor_data = {}
    blobs = sensor_data_client.list_blobs()
    for blob in blobs:
        if blob.name.lower().endswith('.txt'):  # Assumes sensor data is in .txt files
            blob_client = sensor_data_client.get_blob_client(blob)
            data = blob_client.download_blob().readall()
            data_text = data.decode('utf-8')  # Decode byte data to string
            values = data_text.split(',')
            if len(values) == 4:  # Ensure there are exactly four measurements
                sensor_data[blob.name.split('_')[0]] = {
                    'temperature': values[0].strip(),
                    'pressure': values[1].strip(),
                    'humidity': values[2].strip(),
                    'temp_fahrenheit': values[3].strip()
                }
    return sensor_data

# Load sensor data
sensor_data = load_sensor_data(sensor_data_client)

# Streamlit UI
st.title('Farmbeats')

# Display images and corresponding sensor data
st.subheader('Images and Sensor Data from Azure Blob Storage:')
blobs = container_client.list_blobs()
for blob in blobs:
    if blob.name.lower().endswith(('.png', '.jpg', '.jpeg')):
        blob_client = container_client.get_blob_client(blob)
        image = load_image_from_blob(blob_client)
        timestamp = blob.name.split('_')[0]  # Extract timestamp from the image filename
        if timestamp in sensor_data:
            data = sensor_data[timestamp]
            sensor_info = f"Temp: {data['temperature']}°C, Pressure: {data['pressure']} Pa, Humidity: {data['humidity']}%, Temp Fahrenheit: {data['temp_fahrenheit']}°F"
            st.image(image, caption=f"{blob.name} - {sensor_info}", use_column_width=True)
        else:
            st.image(image, caption=f"{blob.name} - No sensor data available", use_column_width=True)
            st.write(f"Debug: No data for timestamp {timestamp}")


