#!/usr/bin/env python3
"""YOLOE Open-Vocabulary Object Detection & Categorization CLI Diagnostic Tool.

Tests model loading, open-vocabulary classification, inference latency, sweet-spot
alignment, and detection accuracy from live camera, synthetic frames, or image files.
"""

import argparse
import os
import sys
import time
import numpy as np
import yaml

# Set up project base path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from src.adapters.vision.yoloe_adapter import YOLOEVisionAdapter, DEFAULT_TARGET_CLASSES
from src.adapters.camera.v4l2_cam import V4L2CameraAdapter
from src.adapters.camera.mock_camera import MockCameraAdapter

VISION_CONFIG_PATH = os.path.join(BASE_DIR, "config", "vision_config.yaml")


def load_vision_config():
    if os.path.exists(VISION_CONFIG_PATH):
        try:
            with open(VISION_CONFIG_PATH, "r") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"[Config] Error loading {VISION_CONFIG_PATH}: {e}")
    return {}


def format_bbox(bbox):
    if not bbox or len(bbox) != 4:
        return "N/A"
    ymin, xmin, ymax, xmax = bbox
    return f"[{xmin}, {ymin}, {xmax}, {ymax}] ({xmax - xmin}x{ymax - ymin}px)"


def draw_annotations(frame, detections, sweet_spot=None):
    """Draws bounding boxes, labels, and sweet-spot crosshair onto frame."""
    import cv2
    vis = frame.copy()

    # Draw Grasp Sweet Spot if provided
    if sweet_spot:
        cx, cy, bw, bh = sweet_spot
        x1 = max(0, cx - bw // 2)
        y1 = max(0, cy - bh // 2)
        x2 = min(vis.shape[1], cx + bw // 2)
        y2 = min(vis.shape[0], cy + bh // 2)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.drawMarker(vis, (cx, cy), (255, 255, 0), cv2.MARKER_CROSS, 20, 2)
        cv2.putText(vis, "GRASP SWEET SPOT", (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

    # Draw Detections
    for det in detections:
        if not getattr(det, "detected", False):
            continue
        bb = getattr(det, "bounding_box", None)
        if not bb or len(bb) != 4:
            continue
        ymin, xmin, ymax, xmax = bb
        pickable = getattr(det, "pickable", False)
        color = (0, 255, 0) if pickable else (0, 165, 255)
        cv2.rectangle(vis, (xmin, ymin), (xmax, ymax), color, 2)

        color_name = getattr(det, "material_color", "")
        color_tag = f" [{color_name}]" if color_name and color_name != "Unknown" else ""
        lbl = f"{det.category}{color_tag} ({det.confidence:.2f})"
        cv2.putText(vis, lbl, (xmin, max(20, ymin - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw centroid
        dcx, dcy = det.center
        cv2.circle(vis, (dcx, dcy), 4, (0, 0, 255), -1)

        # Draw alignment line connecting center crosshair with detected object
        if sweet_spot:
            scx, scy, _, _ = sweet_spot
            dx = dcx - scx
            is_x_centered = abs(dx) <= 20
            col = (0, 255, 0) if is_x_centered else (0, 165, 255)
            cv2.circle(vis, (scx, scy), 5, col, -1)
            cv2.line(vis, (scx, scy), (dcx, dcy), col, 2)
            align_lbl = f"dx={dx:+d}px [X-CENTERED]" if is_x_centered else f"dx={dx:+d}px [{'TURN R' if dx > 0 else 'TURN L'}]"
            cv2.putText(vis, align_lbl, (min(dcx, scx), max(15, min(dcy, scy) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 2)

    return vis


def main():
    parser = argparse.ArgumentParser(description="ErovoutikaGrab YOLOE Vision Test & Diagnostic Suite")
    parser.add_argument("--cam", action="store_true", help="Capture live frame from USB camera (/dev/video0)")
    parser.add_argument("--mock", action="store_true", help="Use synthetic frame (no camera hardware needed)")
    parser.add_argument("--image", type=str, default=None, help="Path to static image file for inference")
    parser.add_argument("--benchmark", type=int, default=0, help="Run N inference iterations to benchmark latency and FPS")
    parser.add_argument("--continuous", action="store_true", help="Run continuous inference loop in terminal")
    parser.add_argument("--show", action="store_true", help="Display live OpenCV window (requires GUI display)")
    parser.add_argument("--save", type=str, default=None, help="Save annotated detection frame to specified image path")
    parser.add_argument("--conf", type=float, default=None, help="Confidence threshold (default: config value or 0.25)")
    parser.add_argument("--imgsz", type=int, default=None, help="Inference image resolution (default: 320)")
    parser.add_argument("--classes", type=str, default=None, help="Comma-separated open-vocabulary target classes")
    parser.add_argument("--model", type=str, default=None, help="Path to YOLOE model weights")
    parser.add_argument("--enforce-limits", action="store_true", help="Enforce size and non-pickable category restrictions (default: False, unconstrained)")
    args = parser.parse_args()

    vcfg = load_vision_config()
    yoloe_cfg = vcfg.get("yoloe", {})
    spot_cfg = vcfg.get("grasp_sweet_spot", {})

    model_path = args.model or yoloe_cfg.get("model_path", "models/yolo11s.tflite")
    conf_thresh = args.conf if args.conf is not None else float(yoloe_cfg.get("confidence_threshold", 0.25))
    imgsz = args.imgsz if args.imgsz is not None else int(yoloe_cfg.get("imgsz", 320))
    enforce_limitations = args.enforce_limits or yoloe_cfg.get("enforce_limitations", False)

    target_classes = yoloe_cfg.get("target_classes", DEFAULT_TARGET_CLASSES)
    if args.classes:
        target_classes = [c.strip() for c in args.classes.split(",") if c.strip()]

    sweet_spot = (
        spot_cfg.get("center_x", 324),
        spot_cfg.get("center_y", 247),
        spot_cfg.get("box_width", 256),
        spot_cfg.get("box_height", 231)
    )

    print("\n" + "=" * 65)
    print("   👁️ EROVOUTIKAGRAB YOLOE VISION DIAGNOSTIC SUITE   ")
    print("=" * 65)
    print(f"Model File       : {model_path}")
    print(f"Confidence Thresh: {conf_thresh:.2f}")
    print(f"Inference Imgsz  : {imgsz}x{imgsz}")
    print(f"Enforce Limits   : {enforce_limitations}")
    print(f"Color Detection  : Enabled (Sub-millisecond HSV)")
    print(f"Target Classes   : {target_classes}")
    print(f"Grasp Sweet Spot : Center=({sweet_spot[0]}, {sweet_spot[1]}), Dim=({sweet_spot[2]}x{sweet_spot[3]})")
    print("=" * 65 + "\n")

    # 1. Initialize Adapter
    print(f"[Init] Instantiating YOLOEVisionAdapter...")
    t0 = time.time()
    adapter = YOLOEVisionAdapter(
        model_path=model_path,
        confidence_threshold=conf_thresh,
        imgsz=imgsz,
        target_classes=target_classes,
        enforce_limitations=enforce_limitations
    )
    init_time = time.time() - t0

    if not adapter.model_loaded:
        print("[Error] Failed to load YOLOE / LiteRT model weights. Check path and torch/ultralytics installation.")
        sys.exit(1)

    print(f"[Init] Model loaded via {adapter.backend_type} and warmed up in {init_time:.2f}s.\n")

    # 2. Acquire Test Frame
    frame = None
    cam = None

    if args.image:
        import cv2
        if not os.path.exists(args.image):
            print(f"[Error] Image file not found: {args.image}")
            sys.exit(1)
        frame = cv2.imread(args.image)
        print(f"[Frame] Loaded image from {args.image} ({frame.shape[1]}x{frame.shape[0]})")
    elif args.cam:
        cam_cfg = vcfg.get("camera", {})
        idx = cam_cfg.get("device_index", 0)
        w = cam_cfg.get("width", 640)
        h = cam_cfg.get("height", 480)
        fps = cam_cfg.get("fps", 30)
        fourcc = cam_cfg.get("fourcc", "MJPG")
        print(f"[Camera] Connecting to /dev/video{idx} ({w}x{h} @ {fps}fps)...")
        cam = V4L2CameraAdapter(device_index=idx, width=w, height=h, target_fps=fps, fourcc=fourcc)
        if not cam.start():
            print("[Warning] Could not open /dev/video0. Falling back to synthetic frame.")
            cam = MockCameraAdapter(width=640, height=480)
            cam.start()
        # Warmup camera
        time.sleep(0.3)
        frame = cam.get_frame()
    else:
        # Default or --mock: create realistic synthetic frame with drawn objects
        print("[Frame] Generating synthetic visual test frame with target object...")
        import cv2
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 40
        # Draw a synthetic bottle-like shape in the sweet spot
        bx, by = sweet_spot[0], sweet_spot[1]
        cv2.rectangle(frame, (bx - 40, by - 80), (bx + 40, by + 80), (180, 220, 240), -1)
        cv2.rectangle(frame, (bx - 20, by - 110), (bx + 20, by - 80), (50, 120, 200), -1)
        cv2.putText(frame, "TARGET", (bx - 30, by), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    if frame is None:
        print("[Error] Could not obtain frame for testing.")
        sys.exit(1)

    # 3. Benchmark Mode
    if args.benchmark > 0:
        n = args.benchmark
        print(f"\n[Benchmark] Running {n} inference passes on {frame.shape[1]}x{frame.shape[0]} frame...")
        latencies = []
        for i in range(n):
            t_start = time.time()
            adapter.categorize(frame)
            elapsed = time.time() - t_start
            latencies.append(elapsed)
            print(f"  Pass {i + 1:2d}/{n}: {elapsed * 1000:6.1f} ms ({1.0 / elapsed:4.1f} FPS)")

        mean_ms = np.mean(latencies) * 1000
        median_ms = np.median(latencies) * 1000
        min_ms = np.min(latencies) * 1000
        max_ms = np.max(latencies) * 1000
        fps = 1000.0 / mean_ms

        print("\n" + "-" * 40)
        print("  📊 INFERENCE BENCHMARK RESULTS")
        print("-" * 40)
        print(f"  Mean Latency   : {mean_ms:6.1f} ms")
        print(f"  Median Latency : {median_ms:6.1f} ms")
        print(f"  Min / Max      : {min_ms:6.1f} ms / {max_ms:6.1f} ms")
        print(f"  Throughput     : {fps:6.1f} FPS")
        print("-" * 40 + "\n")

    # 4. Single Frame Detailed Evaluation
    print("[Inference] Analyzing test frame with YOLOE...")
    t_inf = time.time()
    res = adapter.categorize(frame)
    all_dets = adapter.detect_all(frame)
    inf_elapsed = time.time() - t_inf

    print(f"[Inference] Completed in {inf_elapsed * 1000:.1f} ms (~{1.0 / inf_elapsed:.1f} FPS)")
    print(f"[Results] Total candidate detections: {len(all_dets)}\n")

    if res.detected:
        print("✅ PRIMARY TARGET DETECTED:")
        print(f"   • Category    : {res.category.upper()}")
        print(f"   • Color       : {getattr(res, 'material_color', 'Unknown').upper()}")
        print(f"   • Backend     : {adapter.backend_type}")
        print(f"   • Confidence  : {res.confidence * 100:.1f}%")
        print(f"   • Pickable    : {'YES (Valid Grasp Target)' if res.pickable else 'NO (Rejected/Too Large)'}")
        print(f"   • Bounding Box: {format_bbox(res.bounding_box)}")
        print(f"   • Centroid    : (X={res.center[0]}, Y={res.center[1]})")

        # Alignment Check
        aligned = adapter.verify_grasp_alignment(frame, sweet_spot)
        dx = res.center[0] - sweet_spot[0]
        dy = sweet_spot[1] - res.center[1]
        print(f"   • Sweet Spot  : {'ALIGNED [READY TO GRASP]' if aligned else f'MISALIGNED (dx={dx:+d}px, dy={dy:+d}px)'}")
    else:
        print("ℹ️ NO TARGETS DETECTED in this frame.")

    if len(all_dets) > 1:
        print("\n📋 ALL DETECTIONS FOUND:")
        for idx, d in enumerate(all_dets):
            c_tag = f" [{d.material_color}]" if getattr(d, 'material_color', '') and d.material_color != 'Unknown' else ''
            print(f"   [{idx + 1}] {d.category}{c_tag} ({d.confidence:.2f}) - Box: {format_bbox(d.bounding_box)} - Pickable: {d.pickable}")

    # 5. Save Annotated Output
    if args.save:
        import cv2
        annotated = draw_annotations(frame, all_dets if all_dets else [res], sweet_spot)
        cv2.imwrite(args.save, annotated)
        print(f"\n[Save] Annotated detection frame saved to: {args.save}")

    # 6. Continuous Inference Mode
    if args.continuous and cam:
        import cv2
        print("\n[Continuous] Entering live camera inference loop (Press Ctrl+C to stop)...")
        try:
            while True:
                f = cam.get_frame()
                if f is not None:
                    t_start = time.time()
                    det = adapter.categorize(f)
                    dt = time.time() - t_start
                    stat = f"{det.category.upper()} ({det.confidence * 100:.1f}%)" if det.detected else "SCANNING..."
                    align = "ALIGNED" if adapter.verify_grasp_alignment(f, sweet_spot) else "MISALIGNED"
                    print(f"\r[Live YOLOE] {stat:<26} | SweetSpot: {align:<10} | Latency: {dt * 1000:5.1f}ms ({1.0 / dt:4.1f} FPS)", end="", flush=True)

                    if args.show:
                        vis = draw_annotations(f, [det] if det.detected else [], sweet_spot)
                        cv2.imshow("YOLOE Diagnostic Stream", vis)
                        if cv2.waitKey(1) & 0xFF == 27:
                            break
                time.sleep(0.05)
        except KeyboardInterrupt:
            print("\n[Continuous] Stopped by user.")
        finally:
            if args.show:
                cv2.destroyAllWindows()

    if cam:
        cam.stop()

    print("\n[Done] YOLOE diagnostic completed successfully.")


if __name__ == "__main__":
    main()
