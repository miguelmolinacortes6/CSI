import os
import re
import numpy as np

# =====================================================================
# CONFIGURACIÓN DE RUTAS Y PARÁMETROS
# =====================================================================
input_folder = "/home/user/Desktop/E_CSI_Router/2/laboratorio/training"
output_folder = "/home/user/Desktop/CSI_Procesado/ROUTER/training"
os.makedirs(output_folder, exist_ok=True)

N_SUBPORTADORAS = 64

# Índices estándar de subportadoras nulas/piloto en IEEE 802.11a/g/n (20MHz, 64 FFT)
SUBPORTADORAS_A_BORRAR = {
    0, 1, 2, 3, 4, 5,          # Guarda izquierda
    32,                        # DC (Frecuencia central)
    59, 60, 61, 62, 63,        # Guarda derecha
    11, 25, 39, 53             # Pilotos estándar
}

# =====================================================================
# EXTRAER LABEL DEL NOMBRE DEL ARCHIVO
# =====================================================================
def extract_label(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else -1

# =====================================================================
# SANITIZACIÓN DE FASE (Linear Phase Cleaning)
# =====================================================================
def sanitize_phase(phases):
    unwrapped_phases = np.unwrap(phases)
    indices = np.arange(len(unwrapped_phases))
    A = np.vstack([indices, np.ones(len(indices))]).T
    m, c = np.linalg.lstsq(A, unwrapped_phases, rcond=None)[0]
    sanitized = unwrapped_phases - (m * indices + c)
    return sanitized

# =====================================================================
# PARSER CSI TOLERANTE A ERRORES Y LÍNEAS COMPLEJAS
# =====================================================================
def parse_line(line, n_subportadoras=N_SUBPORTADORAS):
    # Condición de control inicial
    if "CSI_DATA" not in line:
        return None

    try:
        # 1. METADATOS: Buscamos el RSSI (habitualmente el valor negativo tras CSI_DATA,número,MAC)
        # Para evitar problemas con variaciones de columnas, buscamos patrones lógicos de metadatos.
        # Buscamos un número negativo de dos dígitos que suele representar al RSSI (ej: -65)
        rssi_match = re.search(r',(-\d{2}),', line)
        rssi = float(rssi_match.group(1)) if rssi_match else -60.0 # Valor por defecto si falla
        
        # 2. CAPTURA DE DATOS I/Q: Extraemos estrictamente lo que esté dentro de "[ ... ]"
        csi_match = re.search(r'\[(.*?)\]', line)
        if not csi_match:
            return None
            
        csi_raw_str = csi_match.group(1)
        if not csi_raw_str.strip():
            return None
            
        # Convertimos a array de enteros filtrando cualquier espacio o carácter extraño
        data = np.array([int(x) for x in csi_raw_str.split(',') if x.strip() and x.strip() != '-'])
        
        # Filtro estricto ESP32-S3: 64 subportadoras * 2 (I y Q) = 128 elementos
        if len(data) != (n_subportadoras * 2):
            return None
        
        I = data[::2]
        Q = data[1::2]
        
        # 3. PROCESAMIENTO MATEMÁTICO
        amplitude = np.sqrt(I**2 + Q**2)
        amplitude = np.where(amplitude == 0, np.nan, amplitude)
        
        phase_raw = np.arctan2(Q, I)
        phase_sanitized = sanitize_phase(phase_raw)
        
        # 4. REMOCIÓN FÍSICA DE SUBPORTADORAS
        mask_utiles = np.ones(n_subportadoras, dtype=bool)
        mask_utiles[list(SUBPORTADORAS_A_BORRAR)] = False
        
        amplitude_clean = amplitude[mask_utiles]
        phase_clean = phase_sanitized[mask_utiles]
        
        # Unimos las 48 amplitudes útiles + 48 fases útiles + 1 Metadato (RSSI)
        # Total características = 97 columnas puras
        return np.concatenate(([rssi], amplitude_clean, phase_clean))

    except Exception as e:
        # Si algo falla de forma imprevista en la línea, la salta en silencio
        return None

# =====================================================================
# EJECUCIÓN PRINCIPAL
# =====================================================================
for filename in os.listdir(input_folder):
    if filename.endswith(".txt"):
        
        label = extract_label(filename)
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, f"procesado_{filename}")
        
        features_list = []
        
        with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                vec = parse_line(line)
                if vec is not None:
                    features_list.append(vec)
        
        if not features_list:
            print(f"⚠️ Archivo sin muestras válidas: {filename}")
            continue
            
        feat_matrix = np.vstack(features_list)
        
        # Insertamos el Label en la Columna 0 (Total columnas: 1 + 97 = 98 columnas)
        labels_column = np.full((feat_matrix.shape[0], 1), label)
        final_matrix = np.hstack((labels_column, feat_matrix))
        
        final_matrix = np.nan_to_num(final_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Guardar forzando 4 decimales tradicionales sin notación exponencial
        np.savetxt(output_path, final_matrix, delimiter=",", fmt='%.4f')
        
        print(f"✓ Procesado: {filename} → label {label} | Muestras: {final_matrix.shape[0]} | Columnas totales: {final_matrix.shape[1]}")

print("\n🚀 ¡Dataset blindado y procesado con éxito!")