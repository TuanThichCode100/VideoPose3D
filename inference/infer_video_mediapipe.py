# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""Perform inference on videos using MediaPipe Pose, saving outputs
in the Detectron format expected by VideoPose3D (COCO 17-keypoints layout).
"""

import os
import sys
import glob
import time
import argparse
import subprocess as sp
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import mediapipe as mp
except ImportError:
    mp = None


def parse_args():
    parser = argparse.ArgumentParser(description='MediaPipe Pose inference for VideoPose3D')
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for output npz files',
        default='output',
        type=str
    )
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='video extension (default: mp4)',
        default='mp4',
        type=str
    )
    parser.add_argument(
        '--model-complexity',
        dest='model_complexity',
        help='MediaPipe model complexity (0: Lite, 1: Full, 2: Heavy)',
        default=2,
        type=int,
        choices=[0, 1, 2]
    )
    parser.add_argument(
        '--min-detection-confidence',
        dest='min_detection_confidence',
        help='Minimum detection confidence',
        default=0.5,
        type=float
    )
    parser.add_argument(
        '--min-tracking-confidence',
        dest='min_tracking_confidence',
        help='Minimum tracking confidence',
        default=0.5,
        type=float
    )
    parser.add_argument(
        'im_or_folder', help='video file or folder containing videos', default=None
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def read_video_cv2(filename):
    """Fallback reader using OpenCV VideoCapture."""
    cap = cv2.VideoCapture(filename)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {filename}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    cap.release()


def get_resolution(filename):
    command = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height', '-of', 'csv=p=0', filename
    ]
    pipe = sp.Popen(command, stdout=sp.PIPE, stderr=sp.PIPE)
    stdout, _ = pipe.communicate()
    line = stdout.decode().strip()
    if line and ',' in line:
        w, h = line.split(',')
        return int(w), int(h)
    return None, None


def read_video_ffmpeg(filename):
    """Fast stream reading via ffmpeg pipe."""
    w, h = get_resolution(filename)
    if w is None or h is None:
        return None

    command = [
        'ffmpeg',
        '-i', filename,
        '-f', 'image2pipe',
        '-pix_fmt', 'bgr24',
        '-vsync', '0',
        '-vcodec', 'rawvideo', '-'
    ]
    pipe = sp.Popen(command, stdout=sp.PIPE, stderr=sp.DEVNULL, bufsize=-1)
    while True:
        data = pipe.stdout.read(w * h * 3)
        if not data or len(data) < w * h * 3:
            break
        yield np.frombuffer(data, dtype='uint8').reshape((h, w, 3))


def get_video_stream(filename):
    # Try ffmpeg pipe first, fall back to cv2.VideoCapture
    try:
        w, h = get_resolution(filename)
        if w is not None and h is not None:
            return read_video_ffmpeg(filename)
    except Exception:
        pass

    if cv2 is not None:
        return read_video_cv2(filename)
    else:
        raise RuntimeError("Neither working ffmpeg nor opencv-python (cv2) is available.")


# MediaPipe 33 Landmark indices corresponding to 17 COCO keypoints:
# COCO keypoint order:
#  0: nose
#  1: left_eye
#  2: right_eye
#  3: left_ear
#  4: right_ear
#  5: left_shoulder
#  6: right_shoulder
#  7: left_elbow
#  8: right_elbow
#  9: left_wrist
# 10: right_wrist
# 11: left_hip
# 12: right_hip
# 13: left_knee
# 14: right_knee
# 15: left_ankle
# 16: right_ankle
MEDIAPIPE_TO_COCO_INDICES = [
    0,   # 0: nose
    2,   # 1: left_eye
    5,   # 2: right_eye
    7,   # 3: left_ear
    8,   # 4: right_ear
    11,  # 5: left_shoulder
    12,  # 6: right_shoulder
    13,  # 7: left_elbow
    14,  # 8: right_elbow
    15,  # 9: left_wrist
    16,  # 10: right_wrist
    23,  # 11: left_hip
    24,  # 12: right_hip
    25,  # 13: left_knee
    26,  # 14: right_knee
    27,  # 15: left_ankle
    28   # 16: right_ankle
]


def main(args):
    if mp is None:
        raise ImportError(
            "MediaPipe is not installed. Please install it using: pip install mediapipe opencv-python"
        )

    os.makedirs(args.output_dir, exist_ok=True)

    if os.path.isdir(args.im_or_folder):
        im_list = sorted(glob.glob(os.path.join(args.im_or_folder, '*.' + args.image_ext)))
    else:
        im_list = [args.im_or_folder]

    if not im_list:
        print(f"No video files found matching '{args.im_or_folder}' with extension '{args.image_ext}'.")
        return

    mp_pose = mp.solutions.pose

    pose_detector = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=args.model_complexity,
        smooth_landmarks=True,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence
    )

    for video_name in im_list:
        base_name = os.path.basename(video_name)
        out_name = os.path.join(args.output_dir, base_name)
        print(f'Processing {video_name} -> {out_name}.npz')

        boxes = []
        segments = []
        keypoints = []
        im_w, im_h = None, None

        frame_count = 0
        t0 = time.time()

        for frame_i, im in enumerate(get_video_stream(video_name)):
            frame_count += 1
            im_h, im_w = im.shape[:2]

            # MediaPipe expects RGB
            rgb_im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) if cv2 is not None else im[:, :, ::-1]
            results = pose_detector.process(rgb_im)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Convert to (17, 3) [x, y, confidence] in pixel space
                kps_17 = []
                x_coords = []
                y_coords = []

                for idx in MEDIAPIPE_TO_COCO_INDICES:
                    lm = landmarks[idx]
                    x_px = lm.x * im_w
                    y_px = lm.y * im_h
                    conf = lm.visibility
                    kps_17.append([x_px, y_px, conf])
                    x_coords.append(x_px)
                    y_coords.append(y_px)

                kps_17 = np.array(kps_17, dtype=np.float32)  # shape (17, 3)

                # Compute approximate bounding box [x_min, y_min, x_max, y_max, score]
                # with a small padding
                x_min, x_max = max(0.0, float(np.min(x_coords))), min(float(im_w), float(np.max(x_coords)))
                y_min, y_max = max(0.0, float(np.min(y_coords))), min(float(im_h), float(np.max(y_coords)))
                pad_w = 0.1 * (x_max - x_min + 1e-4)
                pad_h = 0.1 * (y_max - y_min + 1e-4)
                bbox = np.array([[
                    max(0.0, x_min - pad_w),
                    max(0.0, y_min - pad_h),
                    min(float(im_w), x_max + pad_w),
                    min(float(im_h), y_max + pad_h),
                    1.0  # Detection score
                ]], dtype=np.float32)

                # Format required by VideoPose3D custom decoder:
                # kps shape: (1, 4, 17) where 4 channels are [x, y, logit(0), prob]
                kps_xy = kps_17[:, :2] # (17, 2)
                kps_prob = kps_17[:, 2:3] # (17, 1)
                kps_logit = np.zeros_like(kps_prob) # Dummy logit
                kps_4 = np.concatenate((kps_xy, kps_logit, kps_prob), axis=1) # (17, 4)
                kps_batch = kps_4[None, ...].transpose(0, 2, 1) # (1, 4, 17)
            else:
                bbox = []
                kps_batch = []

            # Mimic Detectron format expected by VideoPose3D
            cls_boxes = [[], bbox]
            cls_keyps = [[], kps_batch]

            boxes.append(cls_boxes)
            segments.append(None)
            keypoints.append(cls_keyps)

            if frame_i % 30 == 0 and frame_i > 0:
                elapsed = time.time() - t0
                print(f'  Frame {frame_i} ({frame_i / elapsed:.1f} FPS)')

        total_time = time.time() - t0
        print(f'Finished {video_name}: {frame_count} frames in {total_time:.2f}s ({frame_count / max(total_time, 1e-5):.1f} FPS)')

        if im_w is None or im_h is None:
            print(f'Warning: No frames read from {video_name}. Skipping save.')
            continue

        metadata = {
            'w': im_w,
            'h': im_h,
        }

        np.savez_compressed(
            out_name,
            boxes=np.array(boxes, dtype=object),
            segments=np.array(segments, dtype=object),
            keypoints=np.array(keypoints, dtype=object),
            metadata=metadata
        )
        print(f'Saved detections to {out_name}.npz\n')

    pose_detector.close()


if __name__ == '__main__':
    args = parse_args()
    main(args)
