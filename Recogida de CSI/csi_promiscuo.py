import serial
import time
import os

PORT = "/dev/ttyACM0"
BAUDRATE = 921600

OUTPUT_NAME = input("Nombre archivo: ")
DURATION = int(input("Duración (s): "))

OUTPUT_DIR = (
    "/home/user/Desktop/E_CSI_SNIFFER")

os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, OUTPUT_NAME)

ser = serial.Serial(PORT, BAUDRATE, timeout=0)

start_time = time.time()
buffer = ""

with open(OUTPUT_FILE, "w") as f:

    while time.time() - start_time < DURATION:

        data = ser.read(4096).decode(errors="ignore")
        buffer += data

        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.strip()

            if not line:
                continue

            # Guardamos TODO (CSI o RSSI o debug)
            now = time.time()
            ts = time.strftime("%d/%m/%Y %H:%M:%S", time.localtime(now))
            ms = int((now % 1) * 1000)

            f.write(f"{ts}.{ms:03d},{line}\n")
            print(line)

ser.close()
print("Captura finalizada:", OUTPUT_FILE)
