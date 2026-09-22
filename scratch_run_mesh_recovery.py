import os
import sys

WORKSPACE_ROOT = r"e:\BAS\realtime-work-detection"
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.core.mesh_recovery import start_async_mesh_recovery

video_path = r"experiments\video_2026-09-17_18-22-25.mp4"

print(f"Running mesh recovery on {video_path}...")
# We will do it synchronously here since we are in a standalone script
output_dir = "experiments/mesh_recovery"
os.makedirs(output_dir, exist_ok=True)
tmp_dir = os.path.join(output_dir, "tmp")
os.makedirs(tmp_dir, exist_ok=True)

from src.multihmr2 import init_hmr_session, infer_video, render_results_video  # type: ignore

checkpoint_path = "models/hmr2_checkpoints/multihmr2.pt"

print("Initializing HMR session...")
sess = init_hmr_session(checkpoint_path, compile_model=False)
print("HMR session initialized.")
print("Running inference...")
preds = infer_video(sess, video_path, tmp_dir=os.path.join(output_dir, "tmp"))
print("Inference completed.")
print("Rendering results video...")
render_results_video(sess, preds, out_dir=output_dir, tmp_dir=os.path.join(output_dir, "tmp"))
print("Rendering completed.")

print(f"Successfully generated 3D meshes and video overlay in {output_dir}")
