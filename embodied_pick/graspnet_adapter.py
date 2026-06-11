from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

from .schemas import FrameBundle, GraspCandidate, Pose6D


class GraspNetAdapter:
    def __init__(
        self,
        graspnet_root: Path,
        checkpoint_path: Optional[Path],
        dry_run: bool = True,
        collision_thresh: float = 0.01,
        top_k: int = 10,
    ) -> None:
        self.graspnet_root = Path(graspnet_root)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.dry_run = dry_run
        self.collision_thresh = collision_thresh
        self.top_k = top_k
        self._net = None
        self._imports_ready = False

    def predict(self, frame: FrameBundle, target_mask_path: Optional[str] = None) -> List[GraspCandidate]:
        if self.dry_run or not self.checkpoint_path:
            return [
                GraspCandidate(
                    pose=Pose6D(position=[0.25, 0.0, 0.08], quaternion=[0.0, 0.0, 0.0, 1.0], frame_id="camera"),
                    score=0.5,
                    width=0.04,
                    metadata={"mode": "dry_run", "target_mask_path": target_mask_path},
                )
            ]
        self._ensure_imports()
        net = self._get_net()
        end_points, cloud = self._get_and_process_data(frame)
        gg = self._get_grasps(net, end_points)
        if self.collision_thresh > 0:
            gg = self._collision_detection(gg, np.asarray(cloud.points))
        try:
            gg = gg.nms()
        except ImportError as exc:
            print(f"[warning] grasp_nms is not installed, skip NMS: {exc}")
        gg.sort_by_score()
        return self._to_candidates(gg[: self.top_k])

    def _ensure_imports(self) -> None:
        if self._imports_ready:
            return
        api_root = self.graspnet_root.parent / "graspnetAPI"
        if api_root.exists():
            sys.path.insert(0, str(api_root))
        sys.path.insert(0, str(self.graspnet_root / "pointnet2"))
        sys.path.insert(0, str(self.graspnet_root / "knn"))
        sys.path.insert(0, str(self.graspnet_root / "models"))
        sys.path.insert(0, str(self.graspnet_root / "dataset"))
        sys.path.insert(0, str(self.graspnet_root / "utils"))
        self._imports_ready = True

    def _get_net(self):
        if self._net is not None:
            return self._net
        import torch
        from graspnet import GraspNet

        net = GraspNet(
            input_feature_dim=0,
            num_view=300,
            num_angle=12,
            num_depth=4,
            cylinder_radius=0.05,
            hmin=-0.02,
            hmax_list=[0.01, 0.02, 0.03, 0.04],
            is_training=False,
        )
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)
        checkpoint = torch.load(str(self.checkpoint_path), map_location=device)
        net.load_state_dict(checkpoint["model_state_dict"])
        net.eval()
        self._net = net
        return net

    def _get_and_process_data(self, frame: FrameBundle):
        import open3d as o3d
        import scipy.io as scio
        import torch
        from PIL import Image
        from data_utils import CameraInfo, create_point_cloud_from_depth_image

        color = np.asarray(Image.open(frame.rgb_path), dtype=np.float32) / 255.0
        depth_path = frame.depth_path or os.path.join(os.path.dirname(frame.rgb_path), "depth.png")
        mask_path = frame.workspace_mask_path or os.path.join(os.path.dirname(frame.rgb_path), "workspace_mask.png")
        meta_path = frame.meta_path or os.path.join(os.path.dirname(frame.rgb_path), "meta.mat")
        depth = np.asarray(Image.open(depth_path))
        workspace_mask = np.asarray(Image.open(mask_path))
        meta = scio.loadmat(meta_path)
        intrinsic = meta["intrinsic_matrix"]
        factor_depth = meta["factor_depth"]
        camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
        mask = workspace_mask & (depth > 0)
        cloud_masked = cloud[mask]
        color_masked = color[mask]
        num_point = 20000
        if len(cloud_masked) >= num_point:
            idxs = np.random.choice(len(cloud_masked), num_point, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), num_point - len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]
        cloud_o3d = o3d.geometry.PointCloud()
        cloud_o3d.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
        cloud_o3d.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        end_points = {
            "point_clouds": torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32)).to(device),
            "cloud_colors": color_sampled,
        }
        return end_points, cloud_o3d

    @staticmethod
    def _get_grasps(net, end_points):
        import torch
        from graspnet import pred_decode
        from graspnetAPI import GraspGroup

        with torch.no_grad():
            end_points = net(end_points)
            grasp_preds = pred_decode(end_points)
        return GraspGroup(grasp_preds[0].detach().cpu().numpy())

    def _collision_detection(self, gg, cloud_points):
        from collision_detector import ModelFreeCollisionDetector

        detector = ModelFreeCollisionDetector(cloud_points, voxel_size=0.01)
        collision_mask = detector.detect(gg, approach_dist=0.05, collision_thresh=self.collision_thresh)
        return gg[~collision_mask]

    def _to_candidates(self, grasp_group) -> List[GraspCandidate]:
        candidates: List[GraspCandidate] = []
        for grasp in grasp_group:
            rotation = np.asarray(getattr(grasp, "rotation_matrix", np.eye(3)))
            candidates.append(
                GraspCandidate(
                    pose=Pose6D(
                        position=np.asarray(getattr(grasp, "translation", [0, 0, 0]), dtype=float).tolist(),
                        quaternion=_rotation_matrix_to_quaternion(rotation),
                        frame_id="camera",
                    ),
                    score=float(getattr(grasp, "score", 0.0)),
                    width=float(getattr(grasp, "width", 0.0)),
                    depth=float(getattr(grasp, "depth", 0.0)),
                    metadata={"rotation_matrix": rotation.tolist()},
                )
            )
        return candidates


def _rotation_matrix_to_quaternion(matrix: np.ndarray) -> List[float]:
    m = matrix
    trace = float(np.trace(m))
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    else:
        qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0
    return [float(qx), float(qy), float(qz), float(qw)]
