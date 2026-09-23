# Multi-HMR 2
# Copyright (c) 2026-present NAVER Corp.

"""Command-line entry point for the ``multihmr2`` console script."""

import os
import argparse

from .api import (
    init_hmr_session,
    infer_image,
    save_results_image,
    render_results_image,
    infer_video,
    save_results_video,
    render_results_video,
)

def main():
    """Entry point for the multihmr2 CLI.

    Parses command-line arguments, creates an InferenceSession from the given
    checkpoint, and runs inference on either a single image or a video. Optionally
    saves meshes, Anny parameters, and rendered output.
    """
    p = argparse.ArgumentParser("multihmr2")
    p.add_argument("--checkpoint", required=True, help="Path to model checkpoint or folder")
    input_group = p.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", default=None, help="Path to an input image")
    input_group.add_argument("--video", default=None, help="Path to an input video")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Inference device (default: auto; use cuda to require GPU inference)",
    )

    p.add_argument("--tmp_dir", default=None, help="Temp folder for extracted video frames")
    p.add_argument("--conf_thresh", type=float, default=0.4)
    p.add_argument("--dist_thresh_nms", type=float, default=0.25)
    p.add_argument("--lowres", action="store_true")

    p.add_argument("--save_mesh", action="store_true")
    p.add_argument("--save_anny_params", action="store_true")
    p.add_argument("--render", action="store_true")
    p.add_argument("--framerate", type=int, default=30)
    p.add_argument(
        "--compile",
        action="store_true",
        help=(
            "Compile the encoder and HPH decoder with torch.compile for faster "
            "repeated inference. First run will be slow while kernels are compiled. "
            "Use for video or repeated same-resolution image inference."
        ),
    )

    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    session = init_hmr_session(
        args.checkpoint, compile_model=args.compile, device=args.device
    )

    if args.image is not None:
        pred = infer_image(
            session,
            args.image,
            conf_thresh=args.conf_thresh,
            dist_thresh_nms=args.dist_thresh_nms,
            lowres=args.lowres,
        )
        save_results_image(
            session,
            pred,
            args.out,
            save_mesh=args.save_mesh,
            save_anny_params=args.save_anny_params,
            lowres=args.lowres,
        )
        if args.render:
            render_results_image(session, pred, args.image, args.out, lowres=args.lowres)

    if args.video is not None:
        tmp_dir = args.tmp_dir or os.path.join(args.out, "_frames_tmp")
        os.makedirs(tmp_dir, exist_ok=True)

        pred_global = infer_video(
            session,
            args.video,
            tmp_dir,
            conf_thresh=args.conf_thresh,
            dist_thresh_nms=args.dist_thresh_nms,
            lowres=args.lowres,
        )
        save_results_video(
            session,
            pred_global,
            args.out,
            save_mesh=args.save_mesh,
            save_anny_params=args.save_anny_params,
            lowres=args.lowres,
        )
        if args.render:
            render_results_video(session, pred_global, args.out, tmp_dir, framerate=args.framerate, lowres=args.lowres)
