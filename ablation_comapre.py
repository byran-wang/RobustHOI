#!/usr/bin/env python3
"""
Merge nvdiffrast_overlay.mp4 from multiple ablation output directories and
gt_overlay.mp4 side by side with text captions above each column.

Layout: [GT | col1 | col2 | ... | colN] — all 640x480, captioned at 40px height.
"""
import argparse
import os
import subprocess


CAPTION_HEIGHT = 40
FONT_SIZE = 18
VIDEO_SUBPATH = os.path.join("pipeline_joint_opt", "eval_vis", "nvdiffrast_overlay.mp4")


def build_filter_complex(captions):
    n = len(captions)
    parts = []
    for i, cap in enumerate(captions):
        # Escape chars special to ffmpeg drawtext
        safe = cap.replace("\\", "\\\\").replace("'", "’").replace(":", "\\:")
        parts.append(
            f"[{i}]pad=iw:ih+{CAPTION_HEIGHT}:0:{CAPTION_HEIGHT}:color=black,"
            f"drawtext=text='{safe}':fontcolor=white:fontsize={FONT_SIZE}:"
            f"x=(w-text_w)/2:y={CAPTION_HEIGHT // 2 - FONT_SIZE // 2}[v{i}]"
        )
    vstack_inputs = "".join(f"[v{i}]" for i in range(n))
    parts.append(f"{vstack_inputs}hstack=inputs={n}:shortest=1")
    return ";".join(parts)


def merge(video_paths, captions, out_path):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    cmd = ["ffmpeg", "-y"]
    for vp in video_paths:
        cmd += ["-i", vp]
    cmd += ["-filter_complex", build_filter_complex(captions)]
    cmd += ["-c:v", "libx264", "-crf", "23", "-preset", "fast", out_path]

    print("Running:", " ".join(f"'{a}'" if " " in a else a for a in cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Side-by-side ablation comparison video")
    parser.add_argument("--seq", required=True, help="Sequence name, e.g. GSF12")
    parser.add_argument("--output_dirs", nargs="+", required=True,
                        help="Ablation output root dirs; expects {dir}/{seq}/" + VIDEO_SUBPATH)
    parser.add_argument("--captions", nargs="+", required=True,
                        help="Caption label for each output_dir (same order)")
    parser.add_argument("--gt_dir", required=True,
                        help="Dir containing gt_overlay.mp4")
    parser.add_argument("--out_path", required=True,
                        help="Output .mp4 path")
    args = parser.parse_args()

    if len(args.output_dirs) != len(args.captions):
        parser.error(f"--output_dirs ({len(args.output_dirs)}) and "
                     f"--captions ({len(args.captions)}) must have equal length")

    gt_path = os.path.join(args.gt_dir, "gt_overlay.mp4")
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"GT video not found: {gt_path}")

    video_paths = [gt_path]
    captions = ["GT"]

    for d, cap in zip(args.output_dirs, args.captions):
        vp = os.path.join(d, args.seq, VIDEO_SUBPATH)
        if not os.path.exists(vp):
            raise FileNotFoundError(f"Ablation video not found: {vp}")
        video_paths.append(vp)
        captions.append(cap)

    merge(video_paths, captions, args.out_path)
    print(f"Saved: {args.out_path}")


if __name__ == "__main__":
    main()
