import os
from periphery import GPIO
import time

# Define GPIO pins
# with button to resistor to 3V3 on pin 17
button = GPIO("/dev/gpiochip0", 6, "in")

try:
  while True:
    # Read button state (active low with pull-up)
    if button.read():  # Button pressed (high signal)
        time.sleep(0.1)  # Delay for 100 milliseconds
        # Read button state again after the delay
        if button.read():
          print("Button pressed! Shutting down...")
          # Execute the shutdown command with sudo
          os.system("sudo shutdown now")
          # Exit the loop after initiating shutdown
          break
  
except KeyboardInterrupt:
  # Handle Ctrl+C interrupt gracefully
  print("Exiting...")
finally:
  # Ensure resources are released even on errors
  button.close()
