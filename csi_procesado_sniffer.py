import os
import re
import numpy as np

# =====================================================================
# CONFIGURACIÓN DE RUTAS Y PARÁMETROS
# =====================================================================
input_folder = "/home/user/Desktop/E_CSI_SNIFFER/laboratorio/training"
output_folder = "/home/user/Desktop/CSI_Procesado/SNIFFER/laboratorio/training"
os.makedirs(output_folder, exist_ok=True)

N_SUBPORTADORAS = 64

# Índices estándar de subportadoras nulas/piloto en IEEE 802.11a/g/n (20MHz, 64 FFT)
# Guarda externas (-32 a -27 y 28 a 31), DC central (0) y pilotos típicos (-21, -7, 7, 21)
SUBPORTADORAS_A_BORRAR = {
    0, 1, 2, 3, 4, 5,          # Guarda izquierda
    32,                        # DC (Frecuencia central)
    59, 60, 61, 62, 63,        # Guarda derecha
    11, 25, 39, 53             # Pilotos estándar (aproximados según chipset)
}

# =====================================================================
# EXTRAER LABEL DEL NOMBRE DEL ARCHIVO
# =====================================================================
def extract_label(filename):
    match = re.search(r"\d+", filename)
    return int(match.group()) if match else -1

# =====================================================================
# SANITIZACIÓN DE FASE (Linear Phase Cleaning)
# =====================================================================
def sanitize_phase(phases):
    """
    Elimina los desplazamientos STO y SFO mediante desenrollado 
    y ajuste por regresión lineal.
    """
    # 1. Desenvuelto para eliminar saltos artificiales de 2pi
    unwrapped_phases = np.unwrap(phases)
    
    # 2. Regresión lineal para estimar la pendiente del desfase de hardware
    indices = np.arange(len(unwrapped_phases))
    A = np.vstack([indices, np.ones(len(indices))]).T
    
    # Resolver por mínimos cuadrados: phase = m * index + c
    m, c = np.linalg.lstsq(A, unwrapped_phases, rcond=None)[0]
    
    # 3. Restar la rampa lineal estimada para limpiar la fase
    sanitized = unwrapped_phases - (m * indices + c)
    return sanitized

# =====================================================================
# PARSER CSI OPTIMIZADO PARA ESP32-S3
# =====================================================================
def parse_csi_real_line(line, n_subportadoras=N_SUBPORTADORAS):
    parts = line.strip().split(",")
    if len(parts) < 5:
        return None

    try:
        start_idx = 0
        # Detectar si la línea arranca con texto (timestamp)
        if ":" in parts[0] or "/" in parts[0]:
            start_idx = 1

        # METADATOS
        rssi = float(parts[start_idx])
        rate = float(parts[start_idx + 1])
        chan = float(parts[start_idx + 2])
        length = float(parts[start_idx + 3])

        # CSI RAW (Valores I/Q entrelazados)
        csi_raw = [int(x) for x in parts[start_idx + 4:]]
        
        # FILTRO CRUCIAL ESP32-S3: 
        # Cada subportadora tiene 2 valores (I y Q). Si el tamaño de datos 
        # no equivale exactamente a 64 subportadoras, descartamos el paquete.
        if len(csi_raw) != (n_subportadoras * 2):
            return None

        magnitudes = []
        phases_raw = []

        # I/Q -> Magnitud y Fase Cruda
        for i in range(0, len(csi_raw), 2):
            imag = csi_raw[i]
            real = csi_raw[i + 1]

            # Evitar indeterminaciones matemáticas si ambos son cero
            if real == 0 and imag == 0:
                mag = 0.0
                phase = 0.0
            else:
                mag = np.sqrt(real**2 + imag**2)
                phase = np.arctan2(imag, real)

            magnitudes.append(mag)
            phases_raw.append(phase)

        magnitudes = np.array(magnitudes)
        phases_raw = np.array(phases_raw)

        # Sanitizar las fases del paquete antes del filtrado
        phases_sanitized = sanitize_phase(phases_raw)

        # ELIMINACIÓN FÍSICA de subportadoras basura (Guarda/Pilotos)
        # En lugar de ponerlas a 0, creamos una máscara booleana para borrarlas del vector.
        mask_utiles = np.ones(n_subportadoras, dtype=bool)
        mask_utiles[list(SUBPORTADORAS_A_BORRAR)] = False

        magnitudes_clean = magnitudes[mask_utiles]
        phases_clean = phases_sanitized[mask_utiles]

        # CONSTRUCCIÓN DEL VECTOR FINAL HETEROGÉNEO
        # Estructura: [Metadatos (4)] + [Amplitudes Útiles (48)] + [Fases Sanitizadas Útiles (48)]
        # Total columnas: 4 + 48 + 48 = 100 columnas exactas.
        metadatos = [rssi, rate, chan, length]
        vector_final = metadatos + magnitudes_clean.tolist() + phases_clean.tolist()

        return np.array(vector_final)

    except Exception as e:
        return None

# =====================================================================
# PROCESAMIENTO PRINCIPAL
# =====================================================================
for filename in os.listdir(input_folder):
    if filename.endswith(".txt"):
        label = extract_label(filename)
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, f"procesado_{filename}")

        features = []

        with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.strip():
                    vec = parse_csi_real_line(line)
                    if vec is not None:
                        features.append(vec)

        if len(features) == 0:
            print(f"⚠️ Sin datos válidos (HT de 64 subportadoras) en archivo: {filename}")
            continue

        matrix = np.array(features)

        # Control de NaNs de última instancia por seguridad antes del ML
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

        # Añadir la etiqueta (Label) en la primera columna (Columna 0)
        labels = np.full((matrix.shape[0], 1), label)
        final = np.hstack((labels, matrix))

        # GUARDAR EN FORMATO DECIMAL FIJO (Elimina la notación científica)
        # fmt='%.4f' fuerza 4 decimales estables para evitar variaciones de string
        np.savetxt(output_path, final, delimiter=",", fmt="%.4f")

        print(f"✓ Procesado: {filename} → label {label} | Columnas totales: {final.shape[1]} | Muestras válidas: {final.shape[0]}")

print("\n🚀 Dataset CSI robusto y homogéneo listo para algoritmos KNN.")