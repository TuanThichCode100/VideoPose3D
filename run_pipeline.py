"""
Pipeline script for VideoPose3D:
Executes Step 4 (data preparation) and Step 5 (3D inference & visualization/export)
automatically using existing outputs or customizable inputs.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description, env=None):
    print(f"\n==========================================")
    print(f">> {description}")
    print(f"Command: {cmd}")
    print(f"==========================================\n")
    current_env = os.environ.copy()
    if env:
        current_env.update(env)
    ret = subprocess.call(cmd, shell=True, env=current_env)
    if ret != 0:
        print(f"\n[ERROR] Command failed with return code {ret}")
        sys.exit(ret)


def main():
    parser = argparse.ArgumentParser(description="Automate VideoPose3D Pipeline")
    parser.add_argument(
        "--video",
        type=str,
        default="inference/videos/sample.mp4",
        help="Path to the input video file (default: inference/videos/sample.mp4)",
    )
    parser.add_argument(
        "--detections-dir",
        type=str,
        default="inference/output",
        help="Directory containing 2D keypoint npz detections (default: inference/output)",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="myvideos",
        help="Name for custom 2D dataset npz file (default: myvideos)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="pretrained_h36m_detectron_coco.bin",
        help="Checkpoint file in checkpoint/ directory (default: pretrained_h36m_detectron_coco.bin)",
    )
    parser.add_argument(
        "--output-video",
        type=str,
        default="inference/output/output_3d_viz.mp4",
        help="Output rendered 3D animation video (default: inference/output/output_3d_viz.mp4)",
    )
    parser.add_argument(
        "--output-npz",
        type=str,
        default="inference/output/output_3d_positions.npz",
        help="Output 3D coordinates npz file (default: inference/output/output_3d_positions.npz)",
    )
    parser.add_argument(
        "--skip-prep",
        action="store_true",
        help="Skip Step 4 (prepare_data_2d_custom.py) if data_2d_custom_<name>.npz already exists",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Only export 3D positions npz, do not render MP4 video",
    )
    parser.add_argument(
        "--viz-only",
        action="store_true",
        help="Only render MP4 video, do not export 3D positions npz",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to run 3D inference (auto, cuda, cpu). Default auto falls back to cpu if GPU sm is incompatible.",
    )

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    python_exe = sys.executable

    video_path = Path(args.video).resolve()
    if not video_path.exists():
        print(f"[ERROR] Video file not found: {video_path}")
        sys.exit(1)

    video_filename = video_path.name
    custom_dataset_file = project_root / "data" / f"data_2d_custom_{args.dataset_name}.npz"
    ckpt_file = project_root / "checkpoint" / args.checkpoint

    if not ckpt_file.exists():
        print(f"[ERROR] Checkpoint file not found: {ckpt_file}")
        sys.exit(1)

    # ----------------------------------------------------
    # Bước 1: Chuẩn bị 2D Dataset (Step 4 trong INFERENCE.md)
    # ----------------------------------------------------
    if args.skip_prep and custom_dataset_file.exists():
        print(f">> Skipping Step 4: {custom_dataset_file} already exists.")
    else:
        cmd_prep = (
            f'"{python_exe}" prepare_data_2d_custom.py '
            f'-i "{Path(args.detections_dir).resolve()}" '
            f'-o "{args.dataset_name}"'
        )
        os.chdir(project_root / "data")
        run_command(cmd_prep, "Step 1/2: Preparing 2D Dataset (prepare_data_2d_custom.py)")
        os.chdir(project_root)

    # ----------------------------------------------------
    # Bước 2: Chạy 3D Inference & Xuất Tọa Độ / Render (Step 5)
    # ----------------------------------------------------
    # Determine execution device
    cpu_flag = ""
    run_env = {}
    if args.device == "cpu":
        cpu_flag = " --force-cpu"
        run_env["CUDA_VISIBLE_DEVICES"] = "-1"
    elif args.device == "auto":
        # Check if PyTorch GPU is functional or sm capability is incompatible (e.g. RTX 50-series sm_120)
        arch_check_code = (
            "import torch; "
            "cap = f'sm_{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]}'; "
            "archs = torch.cuda.get_arch_list(); "
            "print('COMPATIBLE' if cap in archs else 'INCOMPATIBLE')"
        )
        check_proc = subprocess.run([python_exe, "-c", arch_check_code], capture_output=True, text=True)
        if "INCOMPATIBLE" in check_proc.stdout or check_proc.returncode != 0:
            print("[INFO] GPU architecture (e.g. RTX 50-series) is not yet compiled in this PyTorch build. Automatically running on CPU mode.")
            cpu_flag = " --force-cpu"
            run_env["CUDA_VISIBLE_DEVICES"] = "-1"
        else:
            print("[INFO] GPU is fully compatible. Running on GPU.")

    # Ensure parent output directories exist
    Path(args.output_npz).resolve().parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_video).resolve().parent.mkdir(parents=True, exist_ok=True)

    # 2a. Xuất tọa độ 3D keypoints ra file .npz
    if not args.viz_only:
        cmd_export = (
            f'"{python_exe}" run.py '
            f'-d custom -k {args.dataset_name} '
            f'-arc 3,3,3,3,3 -c checkpoint '
            f'--evaluate {args.checkpoint}{cpu_flag} '
            f'--viz-subject {video_filename} '
            f'--viz-action custom '
            f'--viz-camera 0 '
            f'--viz-export "{args.output_npz}"'
        )
        run_command(cmd_export, "Step 2a: Exporting 3D coordinates to NPZ file", env=run_env)

    # 2b. Render video trực quan hóa 3D
    if not args.export_only:
        cmd_render = (
            f'"{python_exe}" run.py '
            f'-d custom -k {args.dataset_name} '
            f'-arc 3,3,3,3,3 -c checkpoint '
            f'--evaluate {args.checkpoint}{cpu_flag} '
            f'--render '
            f'--viz-subject {video_filename} '
            f'--viz-action custom '
            f'--viz-camera 0 '
            f'--viz-video "{video_path}" '
            f'--viz-output "{args.output_video}" '
            f'--viz-size 6'
        )
        run_command(cmd_render, "Step 2b: Rendering 3D animation video (MP4)", env=run_env)

    print("\n==========================================")
    print(">> ALL STEPS COMPLETED SUCCESSFULLY!")
    if not args.viz_only:
        print(f" -> 3D Coordinates saved to: {Path(args.output_npz).resolve()}")
    if not args.export_only:
        print(f" -> 3D Video rendered at:    {Path(args.output_video).resolve()}")
    print("==========================================\n")


if __name__ == "__main__":
    main()
