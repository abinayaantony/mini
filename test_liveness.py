import cv2
import numpy as np
import onnxruntime as ort

from insightface.app import FaceAnalysis


# ==========================================
# 1. LOAD FACE DETECTION MODEL
# ==========================================

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=-1,
    det_size=(640, 640)
)

print("Face model loaded!")


# ==========================================
# 2. LOAD LIVENESS MODEL
# ==========================================

session = ort.InferenceSession(
    "MiniFASNetV2.onnx",
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print("Liveness model loaded!")


# ==========================================
# 3. CROP FACE USING 2.7 SCALE
# ==========================================

def crop_face(image, bbox):

    h, w = image.shape[:2]

    x1, y1, x2, y2 = bbox

    box_w = x2 - x1
    box_h = y2 - y1

    # MiniFASNetV2 uses 2.7 scale
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
# 4. LIVENESS FUNCTION
# ==========================================

def check_liveness(face_image):

    # Convert to float
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

    # Run ONNX model
    output = session.run(
        [output_name],
        {
            input_name: face_image
        }
    )[0]

    # Softmax
    exp_output = np.exp(
        output - np.max(
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
        probabilities[0][predicted_class]
    )

    return predicted_class, confidence


# ==========================================
# 5. WEBCAM
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("Cannot open webcam")
    exit()

print("Webcam started!")
print("Press Q to quit.")


# ==========================================
# 6. PROCESS FRAMES
# ==========================================

while True:

    success, frame = cap.read()

    if not success:
        break

    # Detect faces
    faces = app.get(frame)

    for face in faces:

        # Get RetinaFace/InsightFace bounding box
        bbox = face.bbox.astype(int)

        x1, y1, x2, y2 = bbox

        # Crop face for liveness
        face_crop = crop_face(
            frame,
            bbox
        )

        # Check liveness
        predicted_class, confidence = (
            check_liveness(face_crop)
        )

        # Class 1 = Real
        if predicted_class == 1:

            label = "REAL"

        else:

            label = "FAKE"


        # Draw face box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Display result
        text = (
            f"{label} "
            f"{confidence:.2f} "
            f"class={predicted_class}"
        )

        cv2.putText(
            frame,
            text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # Print only when needed
        print(
            f"Class: {predicted_class}, "
            f"Confidence: {confidence:.2f}"
        )


    cv2.imshow(
        "Liveness Test",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()

cv2.destroyAllWindows()