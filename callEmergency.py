import os
import time
from RPLCD.i2c import CharLCD

# Initialize LCD
lcd = CharLCD(i2c_expander='PCF8574', address=0x27, port=1,
              cols=20, rows=4, dotsize=8)

def display_message(message):
    """Display a message on the LCD"""
    lcd.clear()
    lcd.cursor_pos = (1, 2)
    lcd.write_string(message)

def call_emergency():
    emergency_number = "112"  # Change to the required emergency number
    display_message("Dialing 112...\nPlease wait")
    print(f"Dialing emergency number: {emergency_number}")

    try:
        os.system(f"echo 'ATD{emergency_number};' > /dev/ttyUSB0")  # Adjust for GSM module
        time.sleep(10)  # Simulated call duration

        display_message("Call Ended")
        time.sleep(2)
    
    except Exception as e:
        display_message("Call Failed")
        print(f"Error in making emergency call: {e}")

if __name__ == "__main__":
    call_emergency()
