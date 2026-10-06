"""Utilidades compartidas para registro y reconocimiento facial con InsightFace + OpenCV."""

import threading
import time
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis

BACKEND_DIR = Path(__file__).resolve().parent.parent
KNOWN_FACES_DIR = BACKEND_DIR / "known_faces"
CACHE_PATH = BACKEND_DIR / "data" / "face_index_cache.npz"

SIMILARITY_THRESHOLD = 0.40  # distancia coseno mínima para considerar una coincidencia

# det_size alto (640) para fotos estáticas de registro: prioriza precisión.
# det_size bajo (320) para el video en vivo: prioriza velocidad.
REGISTER_DET_SIZE = (640, 640)
LIVE_DET_SIZE = (320, 320)

_face_apps = {}


def get_face_app(det_size=LIVE_DET_SIZE):
    """Crea (una sola vez por det_size) la instancia de FaceAnalysis con el modelo buffalo_l."""
    if det_size not in _face_apps:
        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=det_size)
        _face_apps[det_size] = app
    return _face_apps[det_size]


def biggest_of(faces):
    """De una lista de rostros detectados, devuelve el de mayor tamaño (o None si está vacía)."""
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def get_biggest_face(frame, det_size=LIVE_DET_SIZE):
    """Devuelve el rostro de mayor tamaño detectado en el frame, o None si no hay ninguno."""
    return biggest_of(get_face_app(det_size).get(frame))


class AsyncDetector:
    """Corre la detección/reconocimiento en un hilo aparte para no bloquear la cámara.

    El hilo principal solo debe llamar a `submit(frame)` (no bloqueante) y
    `get_results()` para leer la última lista de rostros ya procesada. Así la
    cámara se captura y dibuja siempre a su velocidad máxima; el hilo de
    inferencia procesa el frame más reciente disponible y descarta los que se
    acumulen mientras está ocupado, en vez de ir frame por frame en orden.
    """

    def __init__(self, det_size=LIVE_DET_SIZE):
        self._det_size = det_size
        self._lock = threading.Lock()
        self._pending_frame = None
        self._results = []
        self._version = 0  # se incrementa cada vez que hay un resultado nuevo
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def submit(self, frame):
        with self._lock:
            self._pending_frame = frame

    def get_results(self):
        with self._lock:
            return self._results

    @property
    def version(self):
        """Cambia cada vez que `get_results()` trae una detección nueva (no repetida)."""
        with self._lock:
            return self._version

    def stop(self):
        self._running = False
        self._thread.join(timeout=1)

    def _worker(self):
        app = get_face_app(self._det_size)
        while self._running:
            with self._lock:
                frame = self._pending_frame
                self._pending_frame = None

            if frame is None:
                time.sleep(0.005)
                continue

            faces = app.get(frame)

            with self._lock:
                self._results = faces
                self._version += 1


class OpticalFlowTracker:
    """Sigue una caja delimitadora frame a frame con flujo óptico (Lucas-Kanade).

    Mucho más liviano que los TrackerMIL/CSRT/KCF de OpenCV (que tardan
    decenas de ms por frame): aquí solo se sigue el desplazamiento de un
    puñado de puntos característicos dentro de la caja, suficiente para que
    el recuadro se mueva junto con la cara entre una detección real y la
    siguiente.
    """

    def __init__(self, gray_frame, bbox, max_points=30):
        x1, y1, x2, y2 = bbox
        self._bbox = (float(x1), float(y1), float(x2), float(y2))
        roi = gray_frame[int(y1):int(y2), int(x1):int(x2)]
        pts = cv2.goodFeaturesToTrack(roi, maxCorners=max_points, qualityLevel=0.01, minDistance=5)
        if pts is not None:
            pts[:, 0, 0] += x1
            pts[:, 0, 1] += y1
        self._prev_gray = gray_frame
        self._pts = pts

    @property
    def bbox(self):
        """Caja actual como enteros (x1, y1, x2, y2)."""
        return tuple(int(round(v)) for v in self._bbox)

    def update(self, gray_frame):
        """Avanza el seguimiento un frame. Devuelve False si se perdió el rostro."""
        if self._pts is None or len(self._pts) < 4:
            return False

        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(self._prev_gray, gray_frame, self._pts, None)
        status = status.reshape(-1)
        good_new = new_pts[status == 1].reshape(-1, 2)
        good_old = self._pts[status == 1].reshape(-1, 2)

        if len(good_new) < 4:
            self._pts = None
            return False

        dx, dy = np.median(good_new - good_old, axis=0)
        x1, y1, x2, y2 = self._bbox
        self._bbox = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)
        self._prev_gray = gray_frame
        self._pts = good_new.reshape(-1, 1, 2).astype(np.float32)
        return True


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)
    return float(np.dot(a_norm, b_norm))


def build_index():
    """Recorre known_faces/<nombre>/*.jpg y cachea un embedding por cada foto (sin promediar).

    Guardar cada foto por separado (en vez de un promedio por persona) hace la
    búsqueda más robusta cuando hay pocas fotos o fotos con ángulos/luz distintos:
    en el reconocimiento se compara contra todas y se toma la mejor coincidencia.
    """
    names = []
    embeddings = []

    if not KNOWN_FACES_DIR.exists():
        return names, np.empty((0, 512), dtype=np.float32)

    for person_dir in sorted(KNOWN_FACES_DIR.iterdir()):
        if not person_dir.is_dir():
            continue

        for img_path in sorted(person_dir.glob("*.jpg")):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            face = get_biggest_face(img, det_size=REGISTER_DET_SIZE)
            if face is not None:
                names.append(person_dir.name)
                embeddings.append(face.normed_embedding)

    embeddings = np.array(embeddings, dtype=np.float32) if embeddings else np.empty((0, 512), dtype=np.float32)

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE_PATH, names=np.array(names), embeddings=embeddings)

    return names, embeddings


def load_index(force_rebuild: bool = False):
    """Carga el índice de embeddings desde la caché, o lo reconstruye si no existe."""
    if not force_rebuild and CACHE_PATH.exists():
        data = np.load(CACHE_PATH, allow_pickle=True)
        return list(data["names"]), data["embeddings"]
    return build_index()


def find_match(embedding: np.ndarray, names, embeddings):
    """Devuelve (nombre, similitud) de la mejor coincidencia, o (None, mejor_similitud) si no supera el umbral."""
    if len(names) == 0:
        return None, 0.0

    similarities = [cosine_similarity(embedding, known_emb) for known_emb in embeddings]
    best_idx = int(np.argmax(similarities))
    best_sim = similarities[best_idx]

    if best_sim >= SIMILARITY_THRESHOLD:
        return names[best_idx], best_sim
    return None, best_sim
