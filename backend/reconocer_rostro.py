"""Reconoce rostros en vivo desde la webcam comparándolos contra known_faces/.

Uso:
    python reconocer_rostro.py

Controles:
    r -> reconstruir el índice de embeddings (por si agregaste rostros nuevos)
    q -> salir
"""

import cv2

from util.face_engine import (
    AsyncDetector,
    OpticalFlowTracker,
    load_index,
    find_match,
    SIMILARITY_THRESHOLD,
    LIVE_DET_SIZE,
)


def main():
    names, embeddings = load_index()

    if len(names) == 0:
        print("No hay rostros registrados todavía. Ejecuta primero registrar_rostro.py")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return

    print(f"Reconociendo contra {len(set(names))} persona(s): {sorted(set(names))}")
    print("Presiona 'r' para recargar el índice, 'q' para salir.")

    # La identificación (InsightFace) corre en un hilo aparte y tarda varios
    # frames en completarse. Mientras tanto, cada caja se sigue frame a frame
    # con flujo óptico (barato, en el hilo principal) para que el recuadro se
    # mueva en tiempo real junto con la cara, usando la última identidad
    # reconocida como etiqueta hasta que llegue la siguiente detección.
    detector = AsyncDetector(LIVE_DET_SIZE)
    last_version = -1
    tracked = []  # [{"tracker": OpticalFlowTracker, "label": str, "color": tuple}, ...]

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("No se pudo leer frame de la cámara.")
                break

            detector.submit(frame.copy())
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if detector.version != last_version:
                last_version = detector.version
                tracked = []
                for face in detector.get_results():
                    x1, y1, x2, y2 = face.bbox.astype(int)
                    nombre, similitud = find_match(face.normed_embedding, names, embeddings)

                    if nombre is not None:
                        color = (0, 255, 0)
                        etiqueta = f"{nombre} ({similitud:.2f})"
                    else:
                        color = (0, 0, 255)
                        etiqueta = f"Desconocido ({similitud:.2f})"

                    tracker = OpticalFlowTracker(gray, (x1, y1, x2, y2))
                    tracked.append({"tracker": tracker, "label": etiqueta, "color": color})
            else:
                tracked = [item for item in tracked if item["tracker"].update(gray)]

            for item in tracked:
                x1, y1, x2, y2 = item["tracker"].bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), item["color"], 2)
                cv2.putText(
                    frame,
                    item["label"],
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    item["color"],
                    2,
                )

            cv2.putText(
                frame,
                f"Umbral similitud: {SIMILARITY_THRESHOLD}",
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            cv2.imshow("Reconocimiento facial", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                names, embeddings = load_index(force_rebuild=True)
                print(f"Índice recargado: {len(set(names))} persona(s): {sorted(set(names))}")
    finally:
        detector.stop()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
