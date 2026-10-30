"""Safe access to the packaged GLB avatar rigs and dataset pose adapter."""
from __future__ import annotations

import json
from pathlib import Path
import threading

from .avatar_pose import AvatarPoseService


class AvatarRegistry:
    """Expose the registered GLB assets and the model-independent pose protocol."""

    ALLOWED_MODELS = frozenset({"xbot"})

    def __init__(self, asset_root=None):
        candidate_root = Path(asset_root) if asset_root is not None else Path(__file__).parent / "avatars"
        self.asset_root = candidate_root.resolve() if not candidate_root.is_symlink() else None
        self.config = None
        self._pose_services = {}
        self._pose_lock = threading.RLock()
        if self.asset_root and self.asset_root.is_dir():
            config_path = self.asset_root / "models.json"
            if config_path.is_file() and not config_path.is_symlink():
                candidate = json.loads(config_path.read_text(encoding="utf-8"))
                if candidate.get("schema") == "mmwave-densepose.avatar-model-registry.v1":
                    self.config = candidate

    @property
    def available(self):
        return self.asset_root is not None and self.config is not None

    def _asset_path(self, model_id):
        if not self.available or model_id not in self.ALLOWED_MODELS:
            raise KeyError("Avatar model is unavailable")
        model = self.config.get("models", {}).get(model_id)
        if not isinstance(model, dict) or Path(str(model.get("asset", ""))).suffix.lower() != ".glb":
            raise KeyError("Avatar model is unavailable")
        filename = Path(str(model["asset"]))
        if filename.is_absolute() or filename.name != str(model["asset"]):
            raise KeyError("Avatar model is unavailable")
        candidate = self.asset_root / filename
        path = candidate.resolve(strict=True)
        if self.asset_root not in path.parents or candidate.is_symlink() or not path.is_file():
            raise KeyError("Avatar model is unavailable")
        return model, path

    def list_models(self):
        if not self.available:
            return {"available": False, "models": []}
        rows = []
        for model_id in sorted(self.ALLOWED_MODELS):
            try:
                model, _path = self._asset_path(model_id)
            except (KeyError, OSError, RuntimeError):
                continue
            profile_id = model.get("rig_profile")
            profile = self.config.get("rig_profiles", {}).get(profile_id)
            if not isinstance(profile, dict):
                continue
            rows.append({"id": model_id, "label": model.get("label", model_id),
                         "rig_profile_id": profile_id,
                         "rig_profile": profile,
                         "asset_format": "glb",
                         "asset_url": f"/api/avatar/models/{model_id}/asset"})
        return {"available": bool(rows), "models": rows, "default_model": "xbot"}

    def read_asset(self, model_id):
        _model, path = self._asset_path(str(model_id))
        return path.read_bytes()

    def pose(self, keypoints, *, session_id, source="inference", frame_id=None):
        """Convert one dataset-order 19-joint pose using AvatarPoseService."""
        if not self.available:
            return None
        service_key = (str(session_id), str(source))
        with self._pose_lock:
            service = self._pose_services.get(service_key)
            if service is None:
                service = AvatarPoseService()
                if len(self._pose_services) >= 24:
                    self._pose_services.pop(next(iter(self._pose_services)))
                self._pose_services[service_key] = service
            return service.convert(keypoints, source=source, frame_id=frame_id)
