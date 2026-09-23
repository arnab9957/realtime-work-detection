# Multi-HMR 2
# Copyright (c) 2026-present NAVER Corp.

"""Video frame extraction (via ffmpeg) for in-the-wild inputs."""

import os
import shutil
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True # to avoid "OSError: image file is truncated"
import subprocess
import imageio_ffmpeg


def get_ffmpeg_executable():
    """Return a system FFmpeg executable or the bundled imageio-ffmpeg binary."""
    return shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()

def extract_video(video_path, tmp_folder):
    """Extract all frames of a video to a folder as numbered PNG files.

    Uses ffmpeg to decode the video and write frames as %06d.png. Returns
    the sorted list of extracted frame paths.

    Args:
        video_path: Path to the input video file.
        tmp_folder: Directory where extracted frames are written.

    Returns:
        Sorted list of absolute paths to the extracted PNG files.
    """
    command = [get_ffmpeg_executable(),
            '-y',
            '-i', video_path,
            '-f', 'image2',
            '-v', 'error',
            f'{tmp_folder}/%06d.png']
    subprocess.run(command, check=True)

    fns = [x for x in os.listdir(tmp_folder) if x.lower().endswith('.png')]
    fns = [os.path.join(tmp_folder, x) for x in fns]
    fns.sort()

    return fns