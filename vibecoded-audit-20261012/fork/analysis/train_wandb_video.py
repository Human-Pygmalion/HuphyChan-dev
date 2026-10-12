"""Launch mjlab `train` but ALSO upload each recorded rollout video to wandb.

Why this exists: mjlab's VideoRecorder (src/mjlab/utils/wrappers/video_recorder.py)
only writes the training rollout mp4 to disk (`media.write_video`); it never calls
`wandb.log(wandb.Video(...))`, so with `--video --agent.logger wandb` the videos
appear under logs/.../videos/train/ but NOT on wandb.ai. This wrapper monkeypatches
`VideoRecorder._finish_recording` to log the just-saved mp4 to the active wandb run,
then defers to the normal `mjlab.scripts.train.main()` (all `--video*` / `--agent.*`
CLI flags still work). Core mjlab source is untouched.

Usage (identical to `uv run train`, plus --video):
  uv run python analysis/train_wandb_video.py Mjlab-Velocity-Flat-Pygmalion \
    --video --video-interval 3000 --video-length 300 \
    --env.scene.num-envs 8192 --agent.max-iterations N \
    --agent.resume True --agent.load-run <run> --agent.load-checkpoint model_X.pt \
    --agent.run-name <name> --agent.logger wandb
"""

import os

import mjlab.utils.wrappers.video_recorder as _vr

_orig_finish = _vr.VideoRecorder._finish_recording


def _finish_and_upload(self):
  path = self.current_video_path  # captured before _orig clears it
  _orig_finish(self)
  try:
    import wandb

    if path is not None and os.path.exists(path) and wandb.run is not None:
      wandb.log({"train_rollout": wandb.Video(str(path), format="mp4")})
      print(f"[wandb-video] uploaded {path} -> wandb")
  except Exception as e:  # noqa: BLE001
    print(f"[wandb-video] upload skipped: {e}")


_vr.VideoRecorder._finish_recording = _finish_and_upload

from mjlab.scripts.train import main  # noqa: E402

if __name__ == "__main__":
  main()
