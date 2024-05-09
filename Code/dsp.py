import cv2
import numpy as np
import os
import json

def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    processed_lab = cv2.merge([l, a, b])
    processed_image = cv2.cvtColor(processed_lab, cv2.COLOR_LAB2BGR)
    return processed_image

def detect_and_draw_colors(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([22, 120, 120])
    upper_yellow = np.array([38, 255, 255])
    lower_orange = np.array([10, 150, 150])
    upper_orange = np.array([22, 255, 255])

    mask_yellow = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
    mask_orange = cv2.inRange(hsv_image, lower_orange, upper_orange)
    combined_mask = cv2.bitwise_or(mask_yellow, mask_orange)

    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (255, 0, 0), 4)

    color_pixels = np.count_nonzero(combined_mask)
    total_pixels = image.shape[0] * image.shape[1]
    percentage = (color_pixels / total_pixels) * 100

    return image, percentage

def put_percentage_on_image(image, percentage):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"{percentage:.2f}% color"
    font_scale = 8
    bottom_right_corner = (image.shape[1] - 1500, image.shape[0] - 50)
    cv2.putText(image, text, bottom_right_corner, font, font_scale, (255, 255, 255), 20, cv2.LINE_AA)
    return image

def generate_report(filename, percentage, json_path):
    if percentage > 2:
        status = "It might have stripe rust disease!"
    else:
        status = "Safe!"

    report = {
        "filename": filename,
        "percentage": round(percentage, 2),
        "status": status
    }

    with open(json_path, 'w') as json_file:
        json.dump(report, json_file, indent=4)

def process_single_image(photo_path, save_folder):
    image = cv2.imread(photo_path)
    if image is None:
        print(f"Could not load image: {photo_path}")
        return
    image = apply_clahe(image)
    processed_image, percentage = detect_and_draw_colors(image)
    processed_image = put_percentage_on_image(processed_image, percentage)

    filename = os.path.basename(photo_path)
    save_path = os.path.join(save_folder, filename)
    json_path = os.path.join(save_folder, f"{os.path.splitext(filename)[0]}.json")

    cv2.imwrite(save_path, processed_image)
    generate_report(filename, percentage, json_path)

    print(f"Processed {filename}: {percentage:.2f}% yellow and orange.")

def main():
    photo_path = r"C:\Users\Liliana\Desktop\515\code\photo\strip rust\6.jpg"
    save_directory = r"C:\Users\Liliana\Desktop\515\code\photo\strip rust\Processed"
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    process_single_image(photo_path, save_directory)

if __name__ == '__main__':
    main()
