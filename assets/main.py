import cv2
import numpy as np
import pygame
import os
import threading
from twilio.rest import Client

# --- STABLE TENSORFLOW IMPORTS ---
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions


from dotenv import load_dotenv # Run: pip install python-dotenv
load_dotenv()

# ---------------- CONFIGURATION ----------------
load_dotenv()

TWILIO_SID = os.getenv('TWILIO_SID')
TWILIO_TOKEN = os.getenv('TWILIO_TOKEN')
TWILIO_PHONE = os.getenv('TWILIO_PHONE')
TARGET_PHONE = os.getenv('TARGET_PHONE')

# Safety Check: Stop the code if .env is missing or empty
if not all([TWILIO_SID, TWILIO_TOKEN, TWILIO_PHONE, TARGET_PHONE]):
    print("❌ ERROR: Twilio credentials not found!")
    print("Please create a file named '.env' in this folder with your keys.")
    exit() # This stops the crash and tells you why
SAVE_DIR = "captured_animals"
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------- SOUND SETUP ----------------
pygame.mixer.init()
animal_sounds = {
    "dog": pygame.mixer.Sound("dog.mp3"),
    "cow": pygame.mixer.Sound("cow.mp3"),
    "horse": pygame.mixer.Sound("horse.mp3"),
    "sheep": pygame.mixer.Sound("sheep.mp3"),
    "elephant": pygame.mixer.Sound("elephant.mp3"),
}

# ---------------- MODEL SETUP ----------------
# Load model once
model = MobileNetV2(weights="imagenet")

LABEL_MAP = {
    "dog": ["dog"],
    "cow": ["ox", "cow"],
    "horse": ["horse"],
    "sheep": ["ram", "sheep"],
    "elephant": ["elephant"]
}

# ---------------- FUNCTIONS ----------------
def send_sms_background(animal_name):
    """Sends SMS in a separate thread to prevent video lag."""
    try:
        client = Client(TWILIO_SID, TWILIO_TOKEN)
        message = client.messages.create(
            body=f"⚠️ Animal Alert: A {animal_name} has been detected!",
            from_=TWILIO_PHONE,
            to=TARGET_PHONE
        )
        print(f"SMS Sent successfully: {message.sid}")
    except Exception as e:
        print(f"SMS failed to send: {e}")

# ---------------- CAMERA SETUP ----------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_count = 0
last_detected_animal = None

print("System running... Press 'q' to quit.")

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    display_frame = frame.copy()
    frame_count += 1
    detected_animal = None
    confidence = 0

    # ---------------- PREDICTION (Every 10 frames) ----------------
    if frame_count % 10 == 0:
        # Preprocess frame for MobileNetV2
        img = cv2.resize(frame, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)

        # Run inference
        preds = model.predict(img, verbose=0)
        results = decode_predictions(preds, top=5)[0]

        # Check if detected label matches our map
        for _, label, prob in results:
            label = label.lower()
            for animal, keys in LABEL_MAP.items():
                if any(k in label for k in keys) and prob > 0.30:
                    detected_animal = animal
                    confidence = prob
                    break
            if detected_animal:
                break

    # ---------------- DISPLAY + SOUND + SMS ----------------
    if detected_animal:
        # Draw on screen
        cv2.putText(
            display_frame,
            f"Animal Detected: {detected_animal} ({confidence*100:.1f}%)",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        # Only trigger once per unique detection
        if detected_animal != last_detected_animal:
            # 1. Sound
            if detected_animal in animal_sounds:
                animal_sounds[detected_animal].play()
            
            # 2. Save Image
            img_name = f"{detected_animal}_{frame_count}.jpg"
            cv2.imwrite(os.path.join(SAVE_DIR, img_name), frame)
            
            # 3. SMS (Background Thread)
            threading.Thread(target=send_sms_background, args=(detected_animal,), daemon=True).start()
            
            last_detected_animal = detected_animal
    else:
        last_detected_animal = None

    cv2.imshow("Animal Detection System", display_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ---------------- CLEANUP ----------------
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()