import RPi.GPIO as GPIO
import time
import subprocess
from RPLCD.i2c import CharLCD

# Ensure GPIO is cleaned up before setup
GPIO.cleanup()
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)  # Set mode only after cleanup

# Initialize LCD
lcd = CharLCD(i2c_expander='PCF8574', address=0x27, port=1,
              cols=20, rows=4, dotsize=8)

# Define button pins
BUTTON1 = 23
BUTTON2 = 27  # Emergency Call Button
BUTTON3 = 22  # SOS Message Button
RESET_BUTTON = 17  # Reset button

# Set up GPIO as inputs with pull-up resistors
GPIO.setup(BUTTON1, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(BUTTON2, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(BUTTON3, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(RESET_BUTTON, GPIO.IN, pull_up_down=GPIO.PUD_UP)

def stop_process():
    """Stop the currently running process"""
    global current_process
    if current_process is not None:
        current_process.terminate()
        current_process.wait()
        current_process = None

def button_pressed(channel):
    """Handle button press events"""
    global current_process
    stop_process()  # Stop any running script before starting a new one
    
    if channel == BUTTON1:
        print("Starting Train Mode...")
        current_process = subprocess.Popen(["python", "FinalTrain.py"])
        
    elif channel == BUTTON2:
        print("Calling Emergency Number...")
        current_process = subprocess.Popen(["python", "callEmergency.py"])
        
    elif channel == BUTTON3:
        print("Sending SOS Message...")
        current_process = subprocess.Popen(["python", "sendSOS.py"])
        
    elif channel == RESET_BUTTON:
        print("System Reset")
        stop_process()

# ✅ **Ensure safe edge detection**
try:
    GPIO.add_event_detect(BUTTON1, GPIO.FALLING, callback=button_pressed, bouncetime=300)
    GPIO.add_event_detect(BUTTON2, GPIO.FALLING, callback=button_pressed, bouncetime=300)
    GPIO.add_event_detect(BUTTON3, GPIO.FALLING, callback=button_pressed, bouncetime=300)
    GPIO.add_event_detect(RESET_BUTTON, GPIO.FALLING, callback=button_pressed, bouncetime=300)
except RuntimeError as e:
    print(f"Error adding event detect: {e}. Retrying after cleanup...")
    GPIO.cleanup()
    time.sleep(1)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(BUTTON1, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(BUTTON2, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(BUTTON3, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(RESET_BUTTON, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    GPIO.add_event_detect(BUTTON1, GPIO.FALLING, callback=button_pressed, bouncetime=300)
    GPIO.add_event_detect(BUTTON2, GPIO.FALLING, callback=button_pressed, bouncetime=300)
    GPIO.add_event_detect(BUTTON3, GPIO.FALLING, callback=button_pressed, bouncetime=300)
    GPIO.add_event_detect(RESET_BUTTON, GPIO.FALLING, callback=button_pressed, bouncetime=300)

# Start listening for button presses
print("Controller running. Press Ctrl+C to exit.")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Shutting down...")
    GPIO.cleanup()
finally:
    GPIO.cleanup()

