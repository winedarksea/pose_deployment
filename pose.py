#!/usr/bin/env python3
"""
Pose Estimation on Coral Dev Board

Creates an HTTP server on 8080 to display when diagnostic_mode else sends over UDP

python3 pose.py --diagnostic_mode --disable_cropping --use_thunder
"""
import argparse
import os
import time
import urllib.request
import socket
import threading
import http.server
import socketserver
import cv2
import numpy as np

from pycoral.utils.edgetpu import make_interpreter
# from pycoral.utils.edgetpu import run_inference
from pycoral.adapters import common

###############################################################################
# Configuration and Constants 
###############################################################################

# Movenet single pose models for Edge TPU:
THUNDER_MODEL_FILE = "movenet_single_pose_thunder_ptq_edgetpu.tflite"
THUNDER_MODEL_URL = (
    "https://raw.githubusercontent.com/google-coral/test_data/master/"
    "movenet_single_pose_thunder_ptq_edgetpu.tflite"
)
LIGHTNING_MODEL_FILE = "movenet_single_pose_lightning_ptq_edgetpu.tflite"
LIGHTNING_MODEL_URL = (
    "https://raw.githubusercontent.com/google-coral/test_data/master/"
    "movenet_single_pose_lightning_ptq_edgetpu.tflite"
)

THUNDER_INPUT_SIZE = 256
LIGHTNING_INPUT_SIZE = 192

NUM_KEYPOINTS = 17
MIN_CROP_KEYPOINT_SCORE = 0.2

# Keypoint index => body part
KEYPOINT_DICT = {
    'nose': 0,
    'leftEye': 1,
    'rightEye': 2,
    'leftEar': 3,
    'rightEar': 4,
    'leftShoulder': 5,
    'rightShoulder': 6,
    'leftElbow': 7,
    'rightElbow': 8,
    'leftWrist': 9,
    'rightWrist': 10,
    'leftHip': 11,
    'rightHip': 12,
    'leftKnee': 13,
    'rightKnee': 14,
    'leftAnkle': 15,
    'rightAnkle': 16,
}

# Directory for HTTP diagnostic artifacts
DIAG_DIR = os.path.expanduser("~/diag_out")
os.makedirs(DIAG_DIR, exist_ok=True)

# Create placeholder files so the HTTP server can serve them immediately
for fname in ["raw_frame.jpg", "unnorm_keypoints.jpg", "norm_keypoints.jpg", "runtime_info.txt"]:
    fpath = os.path.join(DIAG_DIR, fname)
    if not os.path.exists(fpath):
        if fname.endswith(".jpg"):
            dummy = np.zeros((10,10,3), dtype=np.uint8)
            cv2.imwrite(fpath, dummy)
        else:
            open(fpath, "w").close()

###############################################################################
# HTTP Server for Diagnostic Mode
###############################################################################

class SimpleDiagnosticHTTPHandler(http.server.SimpleHTTPRequestHandler):
    """
    Minimal auto-refresh page that references images and text from DIAG_DIR.
    We override .translate_path() so the server looks in DIAG_DIR directly.
    """

    def do_GET(self):
        # If root path "/", serve a small auto-refresh HTML page referencing our files
        if self.path in ["/", "/index.html"]:
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()

            # Refresh every 2 seconds
            html = """
            <html>
            <head>
                <meta http-equiv="refresh" content="1" />
                <title>Diagnostic View</title>
            </head>
            <body>
                <h1>Diagnostic Mode (auto-refresh every 1s)</h1>

                <p><b>Raw frame with keypoints:</b></p>
                <img src="/raw_frame.jpg" style="max-width: 640px; border:1px solid #ccc"/><br/>

                <p><b>Unnormalized 'stick figure':</b></p>
                <img src="/unnorm_keypoints.jpg" style="max-width: 320px; border:1px solid #ccc"/><br/>

                <p><b>Normalized 'stick figure':</b></p>
                <img src="/norm_keypoints.jpg" style="max-width: 320px; border:1px solid #ccc"/><br/>

                <h2>Runtime Info</h2>
                <iframe src="/runtime_info.txt" style="width:600px; height:200px;"></iframe>
            </body>
            </html>
            """
            self.wfile.write(html.encode('utf-8'))
        else:
            # Otherwise, serve files from DIAG_DIR
            return super().do_GET()

    def translate_path(self, path):
        """
        By default, SimpleHTTPRequestHandler serves from the current working dir.
        We override this to serve from DIAG_DIR instead.
        """
        local_path = path.lstrip('/')
        return os.path.join(DIAG_DIR, local_path)


def start_http_server(port=8080):
    """
    Starts a simple HTTP server in a background thread, serving DIAG_DIR on port 8080.
    """
    handler = SimpleDiagnosticHTTPHandler
    httpd = socketserver.ThreadingTCPServer(("", port), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    print(f"Diagnostic HTTP server started on port {port}, serving files in {DIAG_DIR}.")


###############################################################################
# Crop region helper functions
###############################################################################

def init_crop_region(image_height, image_width):
    """Defines the default crop region (pads the full image to make it square)."""
    if image_width > image_height:
        box_height = image_width / image_height
        box_width = 1.0
        y_min = (image_height / 2 - image_width / 2) / image_height
        x_min = 0.0
    else:
        box_height = 1.0
        box_width = image_height / image_width
        y_min = 0.0
        x_min = (image_width / 2 - image_height / 2) / image_width

    return {
        'y_min': y_min,
        'x_min': x_min,
        'y_max': y_min + box_height,
        'x_max': x_min + box_width,
        'height': box_height,
        'width': box_width
    }

def full_frame_region(image_height, image_width):
    """
    Returns a crop region that corresponds to the entire frame.
    This is used if --disable_cropping is set.
    """
    return {
        'y_min': 0.0,
        'x_min': 0.0,
        'y_max': 1.0,
        'x_max': 1.0,
        'height': 1.0,
        'width': 1.0
    }

def torso_visible(keypoints):
    """Checks whether there are enough torso keypoints."""
    left_hip_score = keypoints[0,0, KEYPOINT_DICT['leftHip'],2]
    right_hip_score = keypoints[0,0, KEYPOINT_DICT['rightHip'],2]
    left_shoulder_score = keypoints[0,0, KEYPOINT_DICT['leftShoulder'],2]
    right_shoulder_score = keypoints[0,0, KEYPOINT_DICT['rightShoulder'],2]
    return ((left_hip_score > MIN_CROP_KEYPOINT_SCORE or
             right_hip_score > MIN_CROP_KEYPOINT_SCORE) and
            (left_shoulder_score > MIN_CROP_KEYPOINT_SCORE or
             right_shoulder_score > MIN_CROP_KEYPOINT_SCORE))

def determine_torso_and_body_range(keypoints, target_keypoints, center_y, center_x):
    """Calculates the maximum distance from each keypoint to the center location."""
    torso_joints = ['leftShoulder', 'rightShoulder', 'leftHip', 'rightHip']
    max_torso_yrange = 0.0
    max_torso_xrange = 0.0
    for joint in torso_joints:
        dist_y = abs(center_y - target_keypoints[joint][0])
        dist_x = abs(center_x - target_keypoints[joint][1])
        max_torso_yrange = max(max_torso_yrange, dist_y)
        max_torso_xrange = max(max_torso_xrange, dist_x)

    max_body_yrange = 0.0
    max_body_xrange = 0.0
    for joint in KEYPOINT_DICT.keys():
        if keypoints[0,0, KEYPOINT_DICT[joint],2] < MIN_CROP_KEYPOINT_SCORE:
            continue
        dist_y = abs(center_y - target_keypoints[joint][0])
        dist_x = abs(center_x - target_keypoints[joint][1])
        max_body_yrange = max(max_body_yrange, dist_y)
        max_body_xrange = max(max_body_xrange, dist_x)

    return [max_torso_yrange, max_torso_xrange, max_body_yrange, max_body_xrange]

def determine_crop_region(keypoints, image_height, image_width):
    """
    Determines the region to crop for the model to run inference on,
    using the torso-and-body-range logic from the previous frame's keypoints.
    """
    target_keypoints = {}
    for joint in KEYPOINT_DICT.keys():
        y = keypoints[0,0,KEYPOINT_DICT[joint], 0] * image_height
        x = keypoints[0,0,KEYPOINT_DICT[joint], 1] * image_width
        target_keypoints[joint] = [y, x]

    if torso_visible(keypoints):
        center_y = (target_keypoints['leftHip'][0] +
                    target_keypoints['rightHip'][0]) / 2.0
        center_x = (target_keypoints['leftHip'][1] +
                    target_keypoints['rightHip'][1]) / 2.0

        (max_torso_yrange, max_torso_xrange,
         max_body_yrange, max_body_xrange) = determine_torso_and_body_range(
             keypoints, target_keypoints, center_y, center_x)

        crop_length_half = max(
            max_torso_xrange * 1.9,
            max_torso_yrange * 1.9,
            max_body_yrange * 1.2,
            max_body_xrange * 1.2)

        tmp = np.array([
            center_x,
            image_width - center_x,
            center_y,
            image_height - center_y
        ])
        crop_length_half = min(crop_length_half, np.max(tmp))

        crop_corner = [center_y - crop_length_half, center_x - crop_length_half]
        crop_length = crop_length_half * 2

        if (crop_length_half > max(image_width, image_height)/2) or (crop_length <= 0):
            return init_crop_region(image_height, image_width)

        region = {
            'y_min': crop_corner[0] / image_height,
            'x_min': crop_corner[1] / image_width,
            'y_max': (crop_corner[0] + crop_length) / image_height,
            'x_max': (crop_corner[1] + crop_length) / image_width,
            'height': crop_length / image_height,
            'width': crop_length / image_width
        }

        if region['y_min'] >= region['y_max'] or region['x_min'] >= region['x_max']:
            return init_crop_region(image_height, image_width)

        return region
    else:
        return init_crop_region(image_height, image_width)

def crop_and_resize(image, crop_region, crop_size):
    """Crops and resize the image to prepare for model input."""
    image_height, image_width, _ = image.shape
    start_y = int(crop_region['y_min'] * image_height)
    end_y   = int(crop_region['y_max'] * image_height)
    start_x = int(crop_region['x_min'] * image_width)
    end_x   = int(crop_region['x_max'] * image_width)

    start_y = max(0, start_y)
    start_x = max(0, start_x)
    end_y = min(image_height, end_y)
    end_x = min(image_width, end_x)

    if end_y <= start_y or end_x <= start_x:
        return None

    cropped = image[start_y:end_y, start_x:end_x]
    if cropped.shape[0] < 2 or cropped.shape[1] < 2:
        return None

    resized = cv2.resize(cropped, (crop_size[1], crop_size[0]))
    return resized

def run_inference_with_crop(interpreter, image, crop_region, crop_size):
    """
    Runs model inference on the cropped region and adjusts keypoints to original scale.
    
    Returns (keypoints_with_scores, inference_time_ms).
    If cropping fails, returns (None, None).
    """
    image_height, image_width, _ = image.shape
    cropped_image = crop_and_resize(image, crop_region, crop_size)
    if cropped_image is None:
        return None, None

    # BGR->RGB
    input_data = cropped_image[:, :, ::-1]
    # t0 = time.time()
    common.set_input(interpreter, input_data)
    interpreter.invoke()
    # t1 = time.time()

    keypoints_with_scores = common.output_tensor(interpreter, 0).copy()
    keypoints_with_scores = keypoints_with_scores.reshape(1, 1, NUM_KEYPOINTS, 3)

    # Translate keypoints back to full image coords
    for idx in range(NUM_KEYPOINTS):
        keypoints_with_scores[0,0,idx,0] = (
            crop_region['y_min'] * image_height +
            crop_region['height'] * image_height *
            keypoints_with_scores[0,0,idx,0]
        ) / image_height
        keypoints_with_scores[0,0,idx,1] = (
            crop_region['x_min'] * image_width +
            crop_region['width'] * image_width *
            keypoints_with_scores[0,0,idx,1]
        ) / image_width

    # inference_time_ms = (t1 - t0)*1000.0
    inference_time_ms = 0.0
    return keypoints_with_scores, inference_time_ms

###############################################################################
# Normalizing keypoints
###############################################################################

def normalize_keypoints(keypoints_with_scores):
    """
    Scale so leftShoulder(5)->rightShoulder(6) is 100 units, shift so leftShoulder is at (0,0).
    Returns a list of (x, y, score).
    """
    kpts = keypoints_with_scores[0,0,:,:]  # shape (17,3)
    left_shoulder = kpts[5]  # [y, x, score]
    right_shoulder = kpts[6] # [y, x, score]

    ls_x, ls_y = left_shoulder[1], left_shoulder[0]
    rs_x, rs_y = right_shoulder[1], right_shoulder[0]

    dist = np.sqrt((rs_x - ls_x)**2 + (rs_y - ls_y)**2)
    if dist < 1e-6:
        dist = 1.0
    scale = 100.0 / dist

    scaled_points = []
    for i in range(NUM_KEYPOINTS):
        y, x, score = kpts[i]
        shift_x = x - ls_x
        shift_y = y - ls_y
        final_x = scale * shift_x
        final_y = scale * shift_y
        scaled_points.append((final_x, final_y, float(score)))
    return scaled_points

###############################################################################
# Drawing and Diagnostic Helpers
###############################################################################

SKELETON_EDGES = [
    (0,1), (1,3), (0,2), (2,4),        # face/ears
    (5,7), (7,9), (6,8), (8,10),       # arms
    (5,6),                             # shoulders
    (5,11), (6,12), (11,12),          # torso
    (11,13), (13,15), (12,14), (14,16)# legs
]

def draw_keypoints_on_image(frame, keypoints_with_scores):
    """
    Draw circles & skeleton edges on 'frame' for keypoints with confidence > 0.2.
    Returns an annotated copy of 'frame'.
    """
    annotated = frame.copy()
    good_points = []
    for i in range(NUM_KEYPOINTS):
        y, x, score = keypoints_with_scores[0,0,i]
        if score > 0.2:
            good_points.append((i, (int(x), int(y))))

    # Draw skeleton lines
    for (a,b) in SKELETON_EDGES:
        pa = next((pt for (idx,pt) in good_points if idx==a), None)
        pb = next((pt for (idx,pt) in good_points if idx==b), None)
        if pa and pb:
            cv2.line(annotated, pa, pb, (255,255,255), 2)

    # Draw circles
    for (idx,(px,py)) in good_points:
        cv2.circle(annotated, (px, py), 5, (0,255,0), -1)
    return annotated

def draw_pose_stick_figure(kpts, size=(300,300)):
    """
    Renders a 300x300 image with a "stick figure" for the list of (x,y,score).
    We auto-scale to fill the 300x300, preserving aspect ratio.
    """
    valid = [(x,y) for (x,y,s) in kpts if s>0.2]
    bg = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    if not valid:
        return bg

    xs = [p[0] for p in valid]
    ys = [p[1] for p in valid]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    w_range = max_x - min_x
    h_range = max_y - min_y
    if w_range<1e-3: w_range=1.0
    if h_range<1e-3: h_range=1.0

    scale_factor = 0.9 * min(size[0]/w_range, size[1]/h_range)
    offset_x = size[0]/2
    offset_y = size[1]/2

    # Build a dict idx->(disp_x, disp_y)
    disp_pts = {}
    for i,(x,y,sc) in enumerate(kpts):
        if sc>0.2:
            dx = (x - (min_x + w_range/2)) * scale_factor + offset_x
            dy = (y - (min_y + h_range/2)) * scale_factor + offset_y
            disp_pts[i] = (int(dx), int(dy))

    # Draw lines
    for (a,b) in SKELETON_EDGES:
        if a in disp_pts and b in disp_pts:
            cv2.line(bg, disp_pts[a], disp_pts[b], (255,255,255), 2)
    # Draw circles
    for i,(px,py) in disp_pts.items():
        cv2.circle(bg, (px, py), 4, (0,255,0), -1)

    return bg

###############################################################################
# Model utilities
###############################################################################

def ensure_model_files(use_thunder):
    """Checks if model file is present; if not, downloads from URL."""
    if use_thunder:
        model_file = THUNDER_MODEL_FILE
        model_url = THUNDER_MODEL_URL
    else:
        model_file = LIGHTNING_MODEL_FILE
        model_url = LIGHTNING_MODEL_URL

    if not os.path.isfile(model_file):
        print(f"Downloading {model_file} from {model_url} ...")
        urllib.request.urlretrieve(model_url, model_file)
        print("Download complete.")
    else:
        print(f"Found local model file: {model_file}")
    return model_file

###############################################################################
# Sending data over UDP
###############################################################################

def send_keypoints_via_udp(sock, scaled_points, udp_ip, udp_port, model_name):
    """
    Send the normalized keypoints + model name via UDP as CSV with body-part names:
      e.g. "model=movenet_thunder,nose_x=...,nose_y=...,nose_conf=...,..."
    """
    part_names = [
        "nose", "leftEye", "rightEye", "leftEar", "rightEar",
        "leftShoulder", "rightShoulder", "leftElbow", "rightElbow",
        "leftWrist", "rightWrist", "leftHip", "rightHip",
        "leftKnee", "rightKnee", "leftAnkle", "rightAnkle"
    ]
    message_parts = [f"model={model_name}"]
    for i,(xx,yy,cc) in enumerate(scaled_points):
        name = part_names[i]
        message_parts.append(f"{name}_x={xx:.2f}")
        message_parts.append(f"{name}_y={yy:.2f}")
        message_parts.append(f"{name}_conf={cc:.3f}")
    msg = ",".join(message_parts)
    sock.sendto(msg.encode('utf-8'), (udp_ip, udp_port))

###############################################################################
# Main
###############################################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_thunder', action='store_true',
                        help='Use Movenet Thunder (256x256) instead of Lightning (192x192)')
    parser.add_argument('--camera_idx', type=int, default=0,
                        help='Index of the video source (default 0)')
    parser.add_argument('--width', type=int, default=640, help='Camera width')
    parser.add_argument('--height', type=int, default=480, help='Camera height')
    parser.add_argument('--fps', type=int, default=30, help='Camera FPS')
    parser.add_argument('--udp_ip', type=str, default="192.168.0.100",
                        help='IP to send UDP data (ignored if diagnostic_mode)')
    parser.add_argument('--udp_port', type=int, default=9999,
                        help='UDP port (ignored if diagnostic_mode)')
    parser.add_argument('--diagnostic_mode', action='store_true',
                        help='Run a minimal HTTP server and display data, instead of sending UDP')
    parser.add_argument('--disable_cropping', action='store_true',
                        help='If set, never apply torso-based cropping; always use entire frame.')
    args = parser.parse_args()

    # Step 1: Ensure model file
    model_file = ensure_model_files(args.use_thunder)
    model_name = "movenet_thunder" if args.use_thunder else "movenet_lightning"

    # Step 2: Create interpreter
    print(f"Loading {model_file} into Edge TPU interpreter...")
    interpreter = make_interpreter(model_file)
    interpreter.allocate_tensors()
    inference_res = common.input_size(interpreter)  # (256,256) or (192,192)

    # Step 3: Open camera
    print(f"Opening camera index={args.camera_idx}, {args.width}x{args.height}@{args.fps}FPS...")
    cap = cv2.VideoCapture(args.camera_idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    if not cap.isOpened():
        print("ERROR: Cannot open camera.")
        return

    # Step 4: UDP or HTTP
    sock = None
    if not args.diagnostic_mode:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    else:
        start_http_server(port=8080)

    # Step 5: Initialize crop region
    if args.disable_cropping:
        current_crop_region = full_frame_region(args.height, args.width)
    else:
        current_crop_region = init_crop_region(args.height, args.width)

    last_keypoints = None

    print("Starting inference loop. Press Ctrl+C to stop.")
    try:
        while True:
            # Retry read
            frame = None
            for attempt in range(3):
                ret, temp = cap.read()
                if ret:
                    frame = temp
                    break
                time.sleep(0.1)

            if frame is None:
                print("Failed to read frame from camera after 3 attempts. Exiting...")
                break

            # Step 6: If we are not disabling cropping, do iterative approach
            if not args.disable_cropping and last_keypoints is not None:
                current_crop_region = determine_crop_region(last_keypoints, args.height, args.width)

            # Step 7: Run inference on (cropped) image
            keypoints_with_scores, inference_time_ms = run_inference_with_crop(
                interpreter, frame, current_crop_region, (inference_res[1], inference_res[0])
            )
            if keypoints_with_scores is None:
                # fallback (if cropping fails)
                if args.disable_cropping:
                    # always full frame
                    current_crop_region = full_frame_region(args.height, args.width)
                else:
                    current_crop_region = init_crop_region(args.height, args.width)
                continue

            last_keypoints = keypoints_with_scores.copy()

            # Step 8: Normalize
            scaled_points = normalize_keypoints(keypoints_with_scores)

            # Step 9: Send via UDP or do HTTP diag
            if not args.diagnostic_mode:
                # Add model name to the CSV
                send_keypoints_via_udp(sock, scaled_points, args.udp_ip, args.udp_port, model_name)
            else:
                # 1) Annotated raw frame
                annotated = draw_keypoints_on_image(frame, keypoints_with_scores)
                cv2.imwrite(os.path.join(DIAG_DIR, "raw_frame.jpg"), annotated)

                # 2) Unnormalized pose figure
                raw_kpts = keypoints_with_scores[0,0,:,:]  # shape (17,3)
                unnorm_points = []
                for i in range(NUM_KEYPOINTS):
                    yy, xx, sc = raw_kpts[i]
                    unnorm_points.append((xx,yy,sc))
                unnorm_fig = draw_pose_stick_figure(unnorm_points, size=(300,300))
                cv2.imwrite(os.path.join(DIAG_DIR, "unnorm_keypoints.jpg"), unnorm_fig)

                # 3) Normalized pose figure
                norm_fig = draw_pose_stick_figure(scaled_points, size=(300,300))
                cv2.imwrite(os.path.join(DIAG_DIR, "norm_keypoints.jpg"), norm_fig)

                # 4) Runtime info
                sy = int(current_crop_region['y_min']*args.height)
                sx = int(current_crop_region['x_min']*args.width)
                ey = int(current_crop_region['y_max']*args.height)
                ex = int(current_crop_region['x_max']*args.width)

                info_text = (
                    f"Model: {model_name}\n"
                    f"Inference time: {inference_time_ms:.2f} ms\n"
                    f"Crop region: [y=({sy}:{ey}), x=({sx}:{ex})]\n"
                    f"Disable cropping: {args.disable_cropping}\n"
                    "Normalized keypoints:\n"
                )
                for i,(xx,yy,cc) in enumerate(scaled_points):
                    info_text += f"  idx={i} => (x={xx:.2f}, y={yy:.2f}, conf={cc:.2f})\n"

                with open(os.path.join(DIAG_DIR, "runtime_info.txt"), "w") as f:
                    f.write(info_text)

    except KeyboardInterrupt:
        print("\nReceived Ctrl+C. Exiting...")

    cap.release()

if __name__ == "__main__":
    main()
