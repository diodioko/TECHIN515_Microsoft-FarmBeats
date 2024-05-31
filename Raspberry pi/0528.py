import datetime
import cv2
import numpy as np
import os
import json
import smbus2
import logging
from bme280 import BME280
from picamera import PiCamera
from picamera.array import PiRGBArray
from azure.storage.blob import BlobServiceClient
from azure.data.tables import TableServiceClient

# Set up logging
logging.basicConfig(filename='/home/liliana/0524/script.log', level=logging.INFO, format='%(asctime)s %(message)s')

def setup_azure():
    try:
        connect_str = "DefaultEndpointsProtocol=https;AccountName=515team2;AccountKey=+wc53G0GKd551uGI/gn+ow5YcrqralBanMwl+MqJoxReUPwSHwBE6wu4Eoh3awBwxR4za3qlC0hQ+AStlJ2PmA==;EndpointSuffix=core.windows.net"
        blob_service_client = BlobServiceClient.from_connection_string(connect_str)

        raw_container_client = blob_service_client.get_container_client("mile2raw")
        processed_container_client = blob_service_client.get_container_client("mile2processed")

        if not raw_container_client.exists():
            raw_container_client.create_container()
        if not processed_container_client.exists():
            processed_container_client.create_container()

        table_service = TableServiceClient.from_connection_string(connect_str)
        table_client = table_service.get_table_client("mile3")

        return blob_service_client, table_client

    except Exception as e:
        logging.error("Error setting up Azure: %s", e)
        raise

def setup_directories():
    try:
        raw_images_dir = "515/mile3raw"
        processed_images_dir = "515/mile3processed"
        os.makedirs(raw_images_dir, exist_ok=True)
        os.makedirs(processed_images_dir, exist_ok=True)
        return raw_images_dir, processed_images_dir

    except Exception as e:
        logging.error("Error setting up directories: %s", e)
        raise

def setup_camera():
    try:
        camera = PiCamera()
        camera.resolution = (1024, 768)
        return camera

    except Exception as e:
        logging.error("Error setting up camera: %s", e)
        raise

def setup_sensor():
    try:
        bus = smbus2.SMBus(1)
        bme280 = BME280(i2c_dev=bus)
        return bme280

    except Exception as e:
        logging.error("Error setting up sensor: %s", e)
        raise

def read_sensor_data(bme280):
    try:
        temperature = bme280.get_temperature()
        pressure = bme280.get_pressure()
        humidity = bme280.get_humidity()
        temperature_fahrenheit = (temperature * 9/5) + 32
        return round(temperature, 2), round(temperature_fahrenheit, 2), round(pressure, 2), round(humidity, 2)
    
    except Exception as e:
        logging.error("Error reading sensor data: %s", e)
        raise

def apply_clahe(image):
    try:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        processed_lab = cv2.merge([l, a, b])
        processed_image = cv2.cvtColor(processed_lab, cv2.COLOR_LAB2BGR)
        return processed_image
    
    except Exception as e:
        logging.error("Error applying CLAHE: %s", e)
        raise

def detect_and_draw_colors(image):
    try:
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([15, 90, 90])
        upper_yellow = np.array([55, 255, 255])
        lower_orange = np.array([5, 100, 100])
        upper_orange = np.array([22, 255, 255])

        mask_yellow = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
        mask_orange = cv2.inRange(hsv_image, lower_orange, upper_orange)
        combined_mask = cv2.bitwise_or(mask_yellow, mask_orange)

        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (255, 0, 0), 4)

        color_pixels = np.count_nonzero(combined_mask)
        total_pixels = image.shape[0] * image.shape[1]
        percentage = (color_pixels / total_pixels) * 100

        return image, round(percentage, 2)

    except Exception as e:
        logging.error("Error detecting and drawing colors: %s", e)
        raise

def put_percentage_on_image(image, percentage):
    try:
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"{percentage:.2f}% color"
        font_scale = 1
        bottom_right_corner = (50, image.shape[0] - 20)
        cv2.putText(image, text, bottom_right_corner, font, font_scale, (255, 255, 255), 2, cv2.LINE_AA)
        return image

    except Exception as e:
        logging.error("Error putting percentage on image: %s", e)
        raise

def rotate_image(image, angle):
    try:
        if angle == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        else:
            raise ValueError("Only 180 degree rotation is supported.")

    except Exception as e:
        logging.error("Error rotating image: %s", e)
        raise

def capture_and_process_image(camera, bme280, raw_images_dir, processed_images_dir, blob_service_client, table_client):
    try:
        rawCapture = PiRGBArray(camera)
        camera.capture(rawCapture, format="bgr")
        raw_image = rawCapture.array

        rotated_image = rotate_image(raw_image, 180)
        processed_image = apply_clahe(rotated_image.copy())
        processed_image, percentage = detect_and_draw_colors(processed_image)
        processed_image = put_percentage_on_image(processed_image, percentage)

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        raw_filename = f"image_{timestamp}-1.jpg"
        processed_filename = f"image_{timestamp}-2.jpg"

        cv2.imwrite(os.path.join(raw_images_dir, raw_filename), rotated_image)
        cv2.imwrite(os.path.join(processed_images_dir, processed_filename), processed_image)

        upload_blob(blob_service_client, raw_images_dir, raw_filename, "mile3raw")
        upload_blob(blob_service_client, processed_images_dir, processed_filename, "mile3processed")

        temperature, temperature_fahrenheit, pressure, humidity = read_sensor_data(bme280)
        save_to_table(table_client, raw_filename, processed_filename, percentage, temperature, temperature_fahrenheit, pressure, humidity)

        logging.info(f"Processed {processed_filename}: {percentage:.2f}% yellow and orange.")
    
    except Exception as e:
        logging.error("Error capturing and processing image: %s", e)
        raise

def upload_blob(blob_service_client, local_dir, filename, container_name):
    try:
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=filename)
        with open(os.path.join(local_dir, filename), "rb") as data:
            blob_client.upload_blob(data)
            logging.info(f"Uploaded {filename} to Azure Blob Storage in container {container_name}")
    
    except Exception as e:
        logging.error("Error uploading blob: %s", e)
        raise

def extract_datetime_from_filename(filename):
    try:
        datetime_str = filename.split('_')[1]
        datetime_str = '-'.join(datetime_str.split('-')[:2])
        return datetime.datetime.strptime(datetime_str, "%Y%m%d-%H%M%S")
    
    except Exception as e:
        logging.error("Error extracting datetime from filename: %s", e)
        raise

def save_to_table(table_client, raw_filename, processed_filename, percentage, temperature, temperature_fahrenheit, pressure, humidity):
    try:
        status = "YES" if percentage > 5 else "NO"
        date_time = extract_datetime_from_filename(raw_filename)
        entity = {
            "PartitionKey": "ImageInfo",
            "RowKey": raw_filename,
            "RawImageName": raw_filename,
            "ProcessedImageName": processed_filename,
            "Percentage": round(percentage, 2),
            "Status": status,
            "TemperatureC": round(temperature, 2),
            "TemperatureF": round(temperature_fahrenheit, 2),
            "Pressure": round(pressure, 2),
            "Humidity": round(humidity, 2),
            "Date": date_time.strftime("%Y/%m/%d %H:%M:%S"),
            "Timestamp": datetime.datetime.now().isoformat()
        }
        table_client.create_entity(entity)
        logging.info(f"Data saved to Azure Table Storage: {entity}")
    
    except Exception as e:
        logging.error("Error saving to table: %s", e)
        raise

if __name__ == "__main__":
    try:
        blob_service_client, table_client = setup_azure()
        raw_images_dir, processed_images_dir = setup_directories()
        camera = setup_camera()
        bme280 = setup_sensor()
        capture_and_process_image(camera, bme280, raw_images_dir, processed_images_dir, blob_service_client, table_client)
    
    except Exception as e:
        logging.error("Error in main execution: %s", e)
