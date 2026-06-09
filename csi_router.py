import serial
import time
import os

PORT = "/dev/ttyACM0"   
BAUDRATE = 921600       

OUTPUT_NAME = input("Introduce el nombre del archivo (ej: prueba1.csv): ")
DURATION = int(input("Introduce la duración de la captura en segundos: "))

option = input("1- SNIFFER  2- ROUTER: ")

match option:
    case "1":
        OUTPUT_DIR = "/home/user/Desktop/E_CSI_SNIFFER"
    case "2":
        OUTPUT_DIR = "/home/user/Desktop/E_CSI_Router"
    case _:
        print("Opción no válida")
        exit()


os.makedirs(OUTPUT_DIR, exist_ok=True)


OUTPUT_FILE = os.path.join(OUTPUT_DIR, OUTPUT_NAME)

ser = serial.Serial(PORT, BAUDRATE, timeout=1)
start_time = time.time()

with open(OUTPUT_FILE, "w") as f:
    while time.time() - start_time < DURATION:
        line = ser.readline().decode(errors='ignore').strip()
        if "CSI_DATA" in line:
            now = time.time()
            timestamp = time.strftime("%d/%m/%Y %H:%M:%S", time.localtime(now))
            milliseconds = int((now % 1) * 1000)
            timestamp = f"{timestamp}.{milliseconds:03d}"

            f.write(f"{timestamp},{line}\n")
            print(f"{timestamp},{line}\n")

ser.close()
print(f"\nDatos CSI guardados en: {OUTPUT_FILE}")
