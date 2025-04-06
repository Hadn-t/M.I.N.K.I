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

def send_sos():
    emergency_number = "112"  # Change this to a predefined contact
    message = "SOS! Emergency situation detected. Please help!"
    
    display_message("Sending SOS...\nPlease wait")
    print(f"Sending SOS message to {emergency_number}")

    try:
        os.system(f"echo 'AT+CMGF=1' > /dev/ttyUSB0")  # Set text mode
        os.system(f"echo 'AT+CMGS=\"{emergency_number}\"' > /dev/ttyUSB0")
        os.system(f"echo '{message}' > /dev/ttyUSB0")
        os.system("echo -e '\x1A' > /dev/ttyUSB0")  # End SMS with CTRL+Z

        display_message("SOS Sent\nSuccessfully")
        time.sleep(2)
    
    except Exception as e:
        display_message("SOS Failed")
        print(f"Error in sending SOS message: {e}")

if __name__ == "__main__":
    send_sos()
