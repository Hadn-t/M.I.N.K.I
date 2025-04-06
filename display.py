import smbus2
import time
import I2C_LCD_driver

Initialize LCD
lcd = I2C_LCD_driver.lcd()

Display Test Message
lcd.lcd_display_string("Hello, RG2004A!", 1)
lcd.lcd_display_string("Raspberry Pi 4", 2)
lcd.lcd_display_string("Python 3.9.2", 3)
lcd.lcd_display_string("I2C Test OK!", 4)

time.sleep(5)
lcd.lcd_clear()
