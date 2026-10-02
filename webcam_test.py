import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis


# ==========================================
# 1. LOAD REGISTERED EMBEDDINGS
# ==========================================

known_embeddings = np.load(
    "known_embeddings.npy"
)

known_names = np.load(
    "known_names.npy"
)

print("Embeddings loaded:", known_embeddings.shape)
print("Registered persons:", np.unique(known_names))


# ==========================================
# 2. LOAD buffalo_l
# ==========================================

app = FaceAnalysis(
    name="buffalo_l",
    providers=[
        "CPUExecutionProvider"
    ]
)

app.prepare(
    ctx_id=-1,
    det_size=(640, 640)
)

print("Face recognition model loaded!")


# ==========================================
# 3. LOAD MiniFASNetV2
# ==========================================

liveness_session = ort.InferenceSession(
    "MiniFASNetV2.onnx",
    providers=[
        "CPUExecutionProvider"
    ]
)

liveness_input_name = (
    liveness_session
    .get_inputs()[0]
    .name
)

liveness_output_name = (
    liveness_session
    .get_outputs()[0]
    .name
)

print("Liveness model loaded!")


# ==========================================
# 4. COSINE SIMILARITY
# ==========================================

def cosine_similarity(a, b):

    return np.dot(a, b) / (
        np.linalg.norm(a) *
        np.linalg.norm(b)
    )


# ==========================================
# 5. RECOGNIZE FACE
# ==========================================

def recognize_face(face_embedding):

    similarities = []

    for embedding in known_embeddings:

        similarity = cosine_similarity(
            face_embedding,
            embedding
        )

        similarities.append(similarity)

    best_index = np.argmax(
        similarities
    )

    best_similarity = similarities[
        best_index
    ]

    best_name = known_names[
        best_index
    ]

    THRESHOLD = 0.5

    if best_similarity >= THRESHOLD:

        return (
            best_name,
            best_similarity
        )

    else:

        return (
            "UNKNOWN",
            best_similarity
        )


# ==========================================
# 6. CROP FACE FOR LIVENESS
# ==========================================

def crop_face(image, bbox):

    h, w = image.shape[:2]

    x1, y1, x2, y2 = bbox

    box_w = x2 - x1
    box_h = y2 - y1

    scale = min(
        (h - 1) / box_h,
        (w - 1) / box_w,
        2.7
    )

    new_w = box_w * scale
    new_h = box_h * scale

    center_x = x1 + box_w / 2
    center_y = y1 + box_h / 2

    new_x1 = max(
        0,
        int(center_x - new_w / 2)
    )

    new_y1 = max(
        0,
        int(center_y - new_h / 2)
    )

    new_x2 = min(
        w - 1,
        int(center_x + new_w / 2)
    )

    new_y2 = min(
        h - 1,
        int(center_y + new_h / 2)
    )

    cropped = image[
        new_y1:new_y2 + 1,
        new_x1:new_x2 + 1
    ]

    cropped = cv2.resize(
        cropped,
        (80, 80)
    )

    return cropped


# ==========================================
# 7. CHECK LIVENESS
# ==========================================

def check_liveness(face_image):

    face_image = face_image.astype(
        np.float32
    )

    # HWC → CHW
    face_image = np.transpose(
        face_image,
        (2, 0, 1)
    )

    # Add batch dimension
    face_image = np.expand_dims(
        face_image,
        axis=0
    )

    # Run MiniFASNetV2
    output = liveness_session.run(
        [liveness_output_name],
        {
            liveness_input_name:
            face_image
        }
    )[0]

    # Softmax
    exp_output = np.exp(
        output -
        np.max(
            output,
            axis=1,
            keepdims=True
        )
    )

    probabilities = (
        exp_output /
        np.sum(
            exp_output,
            axis=1,
            keepdims=True
        )
    )

    predicted_class = int(
        np.argmax(probabilities)
    )

    confidence = float(
        probabilities[
            0,
            predicted_class
        ]
    )

    # Class 1 = REAL
    if predicted_class == 1:

        result = "REAL"

    else:

        result = "FAKE"

    return result, confidence


# ==========================================
# 8. START WEBCAM
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("Error: Cannot open webcam")
    exit()


print("Webcam started!")
print("Press Q to quit.")


# ==========================================
# 9. PROCESS WEBCAM
# ==========================================

while True:

    success, frame = cap.read()

    if not success:

        print("Error reading frame")
        break


    # Detect faces
    faces = app.get(frame)


    # Process every face
    for face in faces:

        # ----------------------------------
        # GET BOUNDING BOX
        # ----------------------------------

        bbox = face.bbox.astype(int)

        x1, y1, x2, y2 = bbox


        # ----------------------------------
        # LIVENESS
        # ----------------------------------

        face_crop = crop_face(
            frame,
            bbox
        )

        liveness, live_confidence = (
            check_liveness(face_crop)
        )


        # ----------------------------------
        # FACE RECOGNITION
        # ----------------------------------

        embedding = face.embedding

        name, similarity = recognize_face(
            embedding
        )


        # ----------------------------------
        # DISPLAY RESULT
        # ----------------------------------

        text = (
            f"{liveness} | "
            f"{name} | "
            f"{similarity:.2f}"
        )


        # Draw rectangle
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )


        # Display name + REAL/FAKE
        cv2.putText(
            frame,
            text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )


        # Display liveness confidence
        live_text = (
            f"Liveness: "
            f"{live_confidence:.2f}"
        )

        cv2.putText(
            frame,
            live_text,
            (x1, y2 + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )


    # ======================================
    # SHOW WEBCAM
    # ======================================

    cv2.imshow(
        "Unknown Person Detection",
        frame
    )


    # Press Q
    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ==========================================
# 10. RELEASE
# ==========================================

cap.release()

cv2.destroyAllWindows()