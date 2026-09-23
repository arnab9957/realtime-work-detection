# Multi-HMR 2
# Copyright (c) 2026-present NAVER Corp.

"""High-level inference API: ``InferenceSession`` and image/video helpers."""

import os
import copy
import anny
import torch
import pickle
import json
import logging
import subprocess

import numpy as np

from PIL import Image
from pathlib import Path
from anny import Anny

from .models.detr_root_relative.decoder import DecoderOutput
from .tracker import FeatPelvisTracker
from .datasets.itw_video import extract_video, get_ffmpeg_executable
from .datasets.itw_image import preprocess_image, ImagePreproc
from .models.detr_root_relative.multi_hmr import DETR_Root_Relative, DETRConfig
from .utils import (
    render_meshes,
    denormalize_rgb,
    get_best_checkpoint,
    load_from_checkpoint,
    demo_color,
)

logger = logging.getLogger(__name__)
try:
    import trimesh
except ImportError:
    logger.warning("Rendering packages not installed, rendering functions will not work")


CHECKPOINT_URL = (
    "https://download.europe.naverlabs.com/ComputerVision/MultiHMR/multihmr2.pt"
)


def _warn_if_exists(path: str) -> None:
    """Log a warning if the given path already exists."""
    if os.path.exists(path):
        logger.warning("Overwriting existing %s", path)


def _download_checkpoint(ckpt_path: Path) -> None:
    """Download the multihmr2 checkpoint to ckpt_path via wget.

    Creates the parent directory if needed and downloads to a temporary ``.part``
    file first, renaming it into place only on success so an interrupted download
    never leaves a corrupt checkpoint behind.
    """
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = ckpt_path.with_name(ckpt_path.name + ".part")
    logger.warning(
        "Checkpoint not found at %s; downloading from %s", ckpt_path, CHECKPOINT_URL
    )
    subprocess.run(["wget", "-O", tmp_path.as_posix(), CHECKPOINT_URL], check=True)
    tmp_path.replace(ckpt_path)


class InferenceSession(object):
    model: DETR_Root_Relative
    device: torch.device
    use_cuda: bool
    body_model_world: Anny
    img_size: int
    patch_size: int
    preprocessor: ImagePreproc

    current_img_name: str | None = None

    def __init__(
        self,
        ckpt_path: str | Path,
        device: torch.device | str | None = None,
        config: DETRConfig | None = None,
        compile_model: bool = False,
    ):
        """Initialize the inference session.

        Selects device (CUDA if available, else CPU), loads the DETR_Root_Relative
        model from the checkpoint, creates a full-body Anny model in
        root_relative_world parameterization, and sets up the image preprocessor.

        Args:
            ckpt_path: Path to the model checkpoint file. Must be named
                "multihmr2".
            device: Target device. If None, uses CUDA if available, else CPU.
            config: Model configuration. If None, uses a default DETRConfig.
            compile_model: If True, compiles the encoder and HPH decoder
                submodules with torch.compile for faster repeated inference.
        """
        if device is not None:
            device = torch.device(device)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt_path = Path(ckpt_path) if isinstance(ckpt_path, str) else ckpt_path

        assert (
            ckpt_path.stem == "multihmr2"
        ), "For now, we only accept the multihmr2 checkpoint"

        if not ckpt_path.is_file():
            _download_checkpoint(ckpt_path)

        assert (
            ckpt_path.is_file()
        ), f"Checkpoint path {ckpt_path} does not exist or is not a file"

        config = DETRConfig() if config is None else copy.deepcopy(config)

        model = DETR_Root_Relative(config)
        model = model.to(device)

        # enforce Anny to move to device
        if hasattr(model, "body_model") and model.body_model.name == "anny":
            model.body_model = model.body_model.to(device)

        # Reload checkpoint in case of restart
        checkpoint, ckpt_path = get_best_checkpoint(ckpt_path.as_posix())
        load_from_checkpoint(model, checkpoint, ckpt_path)

        model.eval()

        self.model = model
        self.device = device
        self.use_cuda = device.type == "cuda"
        logger.info("Using inference device: %s", self.device)
        if not self.use_cuda:
            logger.warning(
                "CUDA is unavailable to PyTorch; video inference will run on CPU "
                "and may take a long time. Install a CUDA-enabled PyTorch build "
                "for GPU inference."
            )

        ## Anny
        self.body_model_world = anny.create_fullbody_model(
            local_changes=True,
            pose_parameterization="root_relative_world",
            remove_unattached_vertices=False,
        ).to(dtype=torch.float32, device=self.device)

        self.img_size = model.img_size
        self.patch_size = model.patch_size
        self.preprocessor = ImagePreproc(self.img_size, self.patch_size)

        if compile_model:
            logger.warning(
                "torch.compile requested. First inference will be slow (typically "
                "5–30 s for ViT-Large while kernels are compiled). If image "
                "dimensions vary between calls, recompilation is triggered at each "
                "new resolution and incurs the same one-time cost again."
            )
            try:
                self.model.encoder = torch.compile(self.model.encoder)
                self.model.full_body_decoder.decoder = torch.compile(
                    self.model.full_body_decoder.decoder
                )
                logger.info("torch.compile applied to encoder and HPH decoder.")
            except Exception as e:
                logger.warning(
                    "torch.compile() failed, continuing in eager mode: %s", e
                )

    def __call__(
        self,
        img: np.ndarray | str,
        conf_thresh: float = 0.4,
        dist_thresh_nms: float = 0.25,
        lowres: bool = False,
    ) -> DecoderOutput:
        """Run inference on a single image.

        Args:
            img: Either a path to an image file (str) or a numpy array of shape
                (H, W, C). When a path is given, session.current_img_name is set
                to the file stem.
            conf_thresh: Confidence threshold for keeping predicted persons.
            dist_thresh_nms: Distance threshold for 3D pelvis NMS.
            lowres: If True, uses the low-resolution body model.

        Returns:
            DecoderOutput for the single image, moved to CPU.
        """
        if isinstance(img, str):
            self.current_img_name = Path(img).stem
            x = preprocess_image(img)
        else:
            x = self.preprocessor(img).unsqueeze(0)

        with torch.inference_mode():
            with torch.amp.autocast(device_type="cuda", enabled=self.use_cuda):
                pred_list = self.model(
                    x.to(self.device),
                    conf_thresh=conf_thresh,
                    dist_thresh_nms=dist_thresh_nms,
                    lowres=lowres,
                )

        pred_list = [p.to("cpu") for p in pred_list]
        torch.cuda.empty_cache()
        return pred_list[0]


def init_hmr_session(
    pretrained: str | Path,
    compile_model: bool = False,
    device: torch.device | str | None = None,
) -> InferenceSession:
    """Create an InferenceSession on the best available device.

    Args:
        pretrained: Path to the model checkpoint.
        compile_model: If True, compile encoder and HPH decoder with
            torch.compile for faster repeated inference.
        device: Inference device. If None or ``"auto"``, selects CUDA when
            available and otherwise CPU. Use ``"cuda"`` to require GPU inference.

    Returns:
        An InferenceSession loaded on CUDA if available, otherwise CPU.
    """
    if device is None or device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but this PyTorch installation has no CUDA support. "
            "Install a CUDA-enabled PyTorch build in the active virtual environment."
        )
    return InferenceSession(pretrained, device=device, compile_model=compile_model)


def infer_image(
    session: InferenceSession,
    img_path: str,
    conf_thresh: float = 0.4,
    dist_thresh_nms: float = 0.25,
    lowres: bool = False,
) -> DecoderOutput:
    """Run inference on a single image file.

    Args:
        session: The active InferenceSession.
        img_path: Path to the input image.
        conf_thresh: Confidence threshold for keeping predicted persons.
        dist_thresh_nms: Distance threshold for 3D pelvis NMS.
        lowres: If True, uses the low-resolution body model.

    Returns:
        DecoderOutput for the image.
    """
    return session(
        img_path,
        conf_thresh=conf_thresh,
        dist_thresh_nms=dist_thresh_nms,
        lowres=lowres,
    )


def save_results_image(
    session: InferenceSession,
    pred_list: DecoderOutput,
    out_dir: str,
    save_mesh: bool = False,
    save_anny_params: bool = False,
    input_colors: list[tuple[float, float, float, float]] | None = None,
    lowres: bool = False,
) -> int:
    """Save mesh and/or Anny parameter files for a single image's predictions.

    Args:
        session: The active InferenceSession; session.current_img_name is used as
            the output file stem.
        pred_list: DecoderOutput containing the predicted persons.
        out_dir: Root directory where subdirectories "meshes/" and "anny_params/"
            are created.
        save_mesh: If True, exports per-person meshes as .glb files under
            out_dir/meshes/.
        save_anny_params: If True, saves pose and shape parameters as a .pkl file
            under out_dir/anny_params/.
        input_colors: Per-person RGBA colors. If None, colors are taken from
            demo_color.
        lowres: If True, uses the low-resolution body model for face export.

    Returns:
        0 if pred_list is empty, 1 otherwise.
    """
    if len(pred_list) == 0:
        logger.info("The list of results is empty")
        return 0

    body_model = (
        session.model.full_body_decoder.lowres_body_model
        if lowres
        else session.model.full_body_decoder.body_model
    )

    # saving human-centric scene
    if save_mesh:
        mesh_dir = os.path.join(out_dir, "meshes")
        os.makedirs(mesh_dir, exist_ok=True)
        colors = demo_color[: len(pred_list)] if input_colors is None else input_colors

        for j in range(len(pred_list)):
            vertices = pred_list.persons.v3d[j].clone().numpy().reshape(-1, 3)
            faces = body_model.faces.numpy()
            # Create trimesh object
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            mesh.visual.vertex_colors = colors[j]

            # Export to GLB (binary glTF)
            fn = os.path.join(mesh_dir, f"{session.current_img_name}_{j:03d}.glb")
            _warn_if_exists(fn)
            mesh.export(fn)

    if save_anny_params:
        anny_dir = os.path.join(out_dir, "anny_params")
        os.makedirs(anny_dir, exist_ok=True)
        
        # The anny parameters are saved following the 'world' parameterization
        out_dict = {}
        out_dict["K"] = pred_list.K
        out_dict["shape"] = pred_list.persons.shape
        bone_poses = pred_list.persons.bone_poses.clone().float()
        # The translation is included in the pose parameters
        translation = pred_list.persons.transl_pelvis
        bone_poses[:, [0], :3, -1] = translation.unsqueeze(1)
        out_dict["pose_parameters"] = bone_poses

        fn = os.path.join(anny_dir, f"{session.current_img_name}.pkl")
        _warn_if_exists(fn)
        with open(fn, "wb") as f:
            pickle.dump(out_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    return 1


def render_results_image(
    session: InferenceSession,
    pred_list: DecoderOutput,
    img_path: str,
    out_dir: str,
    input_colors=None,
    lowres: bool = False,
):
    """Render the predicted meshes onto the input image and save a side-by-side PNG.

    The output file is saved to out_dir/visu/<session.current_img_name>.png as a
    horizontal concatenation of the original preprocessed image and the rendering.

    Args:
        session: The active InferenceSession; session.current_img_name is used as
            the output file stem.
        pred_list: DecoderOutput containing the predicted persons.
        img_path: Path to the original input image, re-preprocessed for rendering.
        out_dir: Root directory; the rendered image is saved to out_dir/visu/.
        input_colors: Per-person RGBA colors. If None, colors are taken from
            demo_color.
        lowres: If True, uses the low-resolution body model faces.

    Returns:
        1.
    """
    body_model = (
        session.model.full_body_decoder.lowres_body_model
        if lowres
        else session.model.full_body_decoder.body_model
    )

    visu_dir = os.path.join(out_dir, "visu")
    os.makedirs(visu_dir, exist_ok=True)
    # image
    x = preprocess_image(img_path)
    img_array = denormalize_rgb(x[0].cpu().numpy())

    if len(pred_list) == 0:
        logger.info("The list of results is empty")
        pred_rend_array = img_array

    else:
        # pred full
        princpt = pred_list.K[[0, 1], [-1, -1]].clone().numpy()
        focal = pred_list.K[[0, 1], [0, 1]].clone().numpy()
        colors = demo_color[: len(pred_list)] if input_colors is None else input_colors

        pred_verts = [p.reshape(-1, 3) for p in pred_list.persons.v3d]
        pred_faces = [body_model.faces.numpy() for _ in range(len(pred_list))]
        _img_array = img_array.copy()
        pred_rend_array = render_meshes(
            _img_array,
            pred_verts,
            pred_faces,
            {"focal": focal, "princpt": princpt},
            color=colors,
        )

    l_img = [img_array, pred_rend_array]

    img = np.concatenate(l_img, 1)
    fn_out = os.path.join(visu_dir, f"{session.current_img_name}.png")
    _warn_if_exists(fn_out)
    Image.fromarray(img).save(fn_out)

    return 1


def infer_video(
    session: InferenceSession,
    video_path: str,
    tmp_dir: str,
    conf_thresh: float = 0.4,
    dist_thresh_nms: float = 0.25,
    lowres: bool = False,
) -> list[DecoderOutput]:
    """Extract frames from a video and run per-frame inference with identity tracking.

    Frames are extracted to tmp_dir with ffmpeg, then each frame is processed
    sequentially. FeatPelvisTracker assigns consistent track IDs across frames using
    tracking features and pelvis position/orientation. The assigned track_id is
    stored in each frame's DecoderOutput.

    Args:
        session: The active InferenceSession.
        video_path: Path to the input video file.
        tmp_dir: Directory where extracted frames (PNG files) are stored.
        conf_thresh: Confidence threshold for per-frame person detection.
        dist_thresh_nms: Distance threshold for 3D pelvis NMS per frame.
        lowres: If True, uses the low-resolution body model for inference.

    Returns:
        List of DecoderOutput, one per video frame, with track_id populated.
    """
    frame_list = extract_video(video_path, tmp_dir)
    tracker = FeatPelvisTracker()
    tracker.reset()

    global_pred = []

    for time, frame in enumerate(frame_list):
        frame_pred = infer_image(
            session,
            frame,
            conf_thresh=conf_thresh,
            dist_thresh_nms=dist_thresh_nms,
            lowres=lowres,
        )

        if len(frame_pred) > 0:
            trackfeat = frame_pred.persons.trackfeat.clone().float()
            pelvis_xyz = frame_pred.persons.transl_pelvis.clone().float()

            pelvis_ijn = (
                torch.cat(
                    [
                        frame_pred.persons.j2d[:, 0] / session.img_size,
                        torch.log(1 / pelvis_xyz[:, 2:3].clamp_min(1e-5)),
                    ],
                    dim=-1,
                )
                .clone()
                .float()
            )

            pelvis_ori = frame_pred.persons.bone_poses[:, 0, :3, :3].clone().float()

            human_ids = tracker.track_next_frame(
                time, trackfeat, pelvis_xyz, pelvis_ijn, pelvis_ori
            )
        else:
            human_ids = torch.empty((0,), dtype=torch.long)

        assert len(human_ids) == frame_pred.persons.num_person, (
            f"Tracker returned {len(human_ids)} IDs for "
            f"{frame_pred.persons.num_person} persons at frame {time}"
        )
        frame_pred.persons.track_id = human_ids
        global_pred.append(frame_pred)

    return global_pred


def save_results_video(
    session: InferenceSession,
    pred_global: list[DecoderOutput],
    out_dir: str,
    save_mesh: bool = False,
    save_anny_params: bool = False,
    lowres: bool = False,
) -> int:
    """Save mesh and/or Anny parameter files for every frame of a video.

    Iterates through each frame's DecoderOutput, sets session.current_img_name to
    the 1-indexed 6-digit frame number, assigns per-person colors from demo_color
    using track IDs, and calls save_results_image for each frame.

    Args:
        session: The active InferenceSession.
        pred_global: List of per-frame DecoderOutput objects, as returned by
            infer_video.
        out_dir: Root directory for output files.
        save_mesh: If True, exports per-person meshes as .glb files.
        save_anny_params: If True, saves pose and shape parameters as .pkl files.
        lowres: If True, uses the low-resolution body model.

    Returns:
        0 if pred_global is empty, 1 otherwise.
    """
    if len(pred_global) == 0:
        logger.info("The list of results is empty")
        return 0

    for time, frame_pred in enumerate(pred_global):
        session.current_img_name = f"{time + 1:06d}"
        joints_dir = Path(out_dir) / "3d_joints"
        joints_dir.mkdir(parents=True, exist_ok=True)
        joints = frame_pred.persons.j3d.detach().cpu()
        with (joints_dir / f"{session.current_img_name}.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(
                {
                    "frame": time + 1,
                    "joints": joints.tolist(),
                    "shape": list(joints.shape),
                    "num_people": joints.shape[0],
                    "num_joints": joints.shape[1],
                    "coordinates": 3,
                },
                file,
            )
        human_ids = frame_pred.persons.track_id.flatten().tolist()
        colors = [demo_color[person_id] for person_id in human_ids]
        save_results_image(
            session,
            frame_pred,
            out_dir,
            save_mesh=save_mesh,
            save_anny_params=save_anny_params,
            input_colors=colors,
            lowres=lowres,
        )

    return 1


def render_results_video(
    session: InferenceSession,
    pred_global: list[DecoderOutput],
    out_dir: str,
    tmp_dir: str,
    framerate: int = 30,
    lowres: bool = False,
) -> int:
    """Render per-frame overlays and assemble them into an output video with ffmpeg.

    For each frame, calls render_results_image using the corresponding extracted
    frame image from tmp_dir, writing rendered PNGs to out_dir/visu/. After all
    frames are rendered, ffmpeg combines them into out_dir/output.mp4 at the
    specified framerate.

    Args:
        session: The active InferenceSession.
        pred_global: List of per-frame DecoderOutput objects, as returned by
            infer_video.
        out_dir: Root directory; rendered PNGs go to out_dir/visu/.
        tmp_dir: Directory containing extracted frames as numbered .png files.
        framerate: Output video framerate in frames per second.
        lowres: If True, uses the low-resolution body model faces.

    Returns:
        0 if pred_global is empty, 1 otherwise.
    """
    if len(pred_global) == 0:
        logger.info("The list of results is empty")
        return 0

    for time, frame_pred in enumerate(pred_global):
        session.current_img_name = f"{time + 1:06d}"
        human_ids = frame_pred.persons.track_id.flatten().tolist()
        colors = [demo_color[person_id] for person_id in human_ids]
        render_results_image(
            session,
            frame_pred,
            os.path.join(tmp_dir, session.current_img_name + ".png"),
            out_dir,
            input_colors=colors,
            lowres=lowres,
        )

    output_video = os.path.join(out_dir, "output.mp4")
    _warn_if_exists(output_video)

    command = [
        get_ffmpeg_executable(),
        "-y",
        "-framerate",
        f"{framerate}",
        "-i",
        f'{os.path.join(out_dir, "visu")}/%06d.png',
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        output_video,
    ]

    subprocess.run(command, check=True)

    return 1
