"""Captura fotos de un rostro desde la webcam y las guarda en known_faces/<nombre>/.

Uso:
    python registrar_rostro.py "Nombre Apellido"

Controles:
    ESPACIO  -> iniciar/detener ráfaga de captura automática (mueve la cabeza
                lentamente -izq/der/arriba/abajo- mientras capturas para variedad)
    q        -> salir
"""

import sys
import time

import cv2

from util.face_engine import (
    KNOWN_FACES_DIR,
    CACHE_PATH,
    AsyncDetector,
    OpticalFlowTracker,
    biggest_of,
    LIVE_DET_SIZE,
)

FOTOS_OBJETIVO = 25
INTERVALO_CAPTURA = 0.25  # segundos entre fotos durante la ráfaga


def main():
    if len(sys.argv) < 2:
        print('Uso: python registrar_rostro.py "Nombre Apellido"')
        sys.exit(1)

    nombre = sys.argv[1].strip().replace(" ", "_")
    person_dir = KNOWN_FACES_DIR / nombre
    person_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        sys.exit(1)

    fotos_tomadas = len(list(person_dir.glob("*.jpg")))
    capturando = False
    ultima_captura = 0.0

    print(f"Registrando a '{nombre}'. Fotos existentes: {fotos_tomadas}")
    print(f"Objetivo recomendado: {FOTOS_OBJETIVO}+ fotos")
    print("Presiona ESPACIO para iniciar/detener la ráfaga, 'q' para salir.")

    # La detección corre en un hilo aparte: la cámara nunca espera al modelo,
    # así el preview se ve fluido a la velocidad real de la webcam. El recuadro
    # se sigue con flujo óptico entre detecciones para que se mueva en tiempo
    # real junto con la cara, no solo cuando llega un resultado nuevo.
    detector = AsyncDetector(LIVE_DET_SIZE)
    last_version = -1
    tracker = None

    while True:
        ok, frame = cap.read()
        if not ok:
            print("No se pudo leer frame de la cámara.")
            break

        detector.submit(frame.copy())
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if detector.version != last_version:
            last_version = detector.version
            face = biggest_of(detector.get_results())
            if face is not None:
                x1, y1, x2, y2 = face.bbox.astype(int)
                tracker = OpticalFlowTracker(gray, (x1, y1, x2, y2))
            else:
                tracker = None
        elif tracker is not None and not tracker.update(gray):
            tracker = None

        display = frame.copy()
        hay_rostro = tracker is not None

        if hay_rostro:
            x1, y1, x2, y2 = tracker.bbox
            color = (0, 255, 0) if not capturando else (0, 150, 255)
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

        if capturando and hay_rostro and (time.time() - ultima_captura) >= INTERVALO_CAPTURA:
            filename = person_dir / f"{int(time.time() * 1000)}.jpg"
            cv2.imwrite(str(filename), frame)
            fotos_tomadas += 1
            ultima_captura = time.time()
            if fotos_tomadas >= FOTOS_OBJETIVO:
                capturando = False
                print(f"Objetivo alcanzado: {fotos_tomadas} fotos.")

        if capturando:
            estado = "CAPTURANDO - mueve la cabeza lentamente (izq/der/arriba/abajo)"
        elif hay_rostro:
            estado = "Listo (ESPACIO para iniciar ráfaga)"
        else:
            estado = "No se detecta rostro"

        cv2.putText(display, estado, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(
            display,
            f"Fotos: {fotos_tomadas}/{FOTOS_OBJETIVO}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Registrar rostro", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord(" "):
            capturando = not capturando

    detector.stop()
    cap.release()
    cv2.destroyAllWindows()

    if fotos_tomadas > 0:
        CACHE_PATH.unlink(missing_ok=True)
        print("Caché de embeddings invalidada. Se reconstruirá en la próxima ejecución de reconocer_rostro.py")

    if fotos_tomadas < FOTOS_OBJETIVO:
        print(f"Aviso: se recomienda capturar al menos {FOTOS_OBJETIVO} fotos para mejorar la precisión.")


if __name__ == "__main__":
    main()
