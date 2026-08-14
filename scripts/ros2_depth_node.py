#!/usr/bin/env python3
"""Publish Fast-FoundationStereo depth from a rectified ROS 2 stereo pair."""

from __future__ import annotations

import copy
import math
import os
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
from core.utils.utils import InputPadder  # noqa: E402

# PyTorch and the model's torch.compile decorators must initialize before
# ROS/TF2/OpenCV native extensions. Reversing this order can crash Torch.
import cv2
import message_filters
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField

import tf2_ros  # noqa: E402

AMP_DTYPE = torch.float16


class FastFoundationStereoDepthNode(Node):
    def __init__(self) -> None:
        super().__init__("fast_foundation_stereo_depth")

        default_model = REPO_ROOT / "weights/23-36-37/model_best_bp2_serialize.pth"
        self.declare_parameter("model_path", str(default_model))
        self.declare_parameter("left_image_topic", "/camera/camera/infra1/image_rect_raw")
        self.declare_parameter("right_image_topic", "/camera/camera/infra2/image_rect_raw")
        self.declare_parameter("left_camera_info_topic", "/camera/camera/infra1/camera_info")
        self.declare_parameter("right_camera_info_topic", "/camera/camera/infra2/camera_info")
        self.declare_parameter("color_image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("color_camera_info_topic", "/camera/camera/color/camera_info")
        self.declare_parameter("depth_topic", "/fast_foundation_stereo/depth/image_raw")
        self.declare_parameter("depth_camera_info_topic", "/fast_foundation_stereo/depth/camera_info")
        self.declare_parameter("pointcloud_topic", "/fast_foundation_stereo/points")
        self.declare_parameter("valid_iters", 4)
        self.declare_parameter("max_disp", 192)
        self.declare_parameter("max_depth", 100.0)
        self.declare_parameter("sync_slop", 0.01)
        self.declare_parameter("max_color_age", 0.05)
        self.declare_parameter("pointcloud_stride", 2)
        self.declare_parameter("build_volume_backend", "pytorch1")

        self.valid_iters = int(self.get_parameter("valid_iters").value)
        self.max_disp = int(self.get_parameter("max_disp").value)
        self.max_depth = float(self.get_parameter("max_depth").value)
        self.backend = str(self.get_parameter("build_volume_backend").value)
        self.max_color_age = float(self.get_parameter("max_color_age").value)
        self.cloud_stride = max(1, int(self.get_parameter("pointcloud_stride").value))
        if self.backend not in ("pytorch1", "triton"):
            raise ValueError("build_volume_backend must be 'pytorch1' or 'triton'")

        self.model = self._load_model(Path(self.get_parameter("model_path").value))
        self.left_info: CameraInfo | None = None
        self.right_info: CameraInfo | None = None
        self.color_info: CameraInfo | None = None
        self.latest_color: Image | None = None
        self._latest_pair: tuple[Image, Image] | None = None
        self._pair_lock = threading.Lock()
        self._pair_ready = threading.Event()
        self._stop = threading.Event()
        self._frames_received = 0
        self._frames_published = 0

        left_info_topic = str(self.get_parameter("left_camera_info_topic").value)
        right_info_topic = str(self.get_parameter("right_camera_info_topic").value)
        self.create_subscription(CameraInfo, left_info_topic, self._on_left_info, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, right_info_topic, self._on_right_info, qos_profile_sensor_data)
        self.create_subscription(
            CameraInfo,
            str(self.get_parameter("color_camera_info_topic").value),
            self._on_color_info,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Image,
            str(self.get_parameter("color_image_topic").value),
            self._on_color,
            qos_profile_sensor_data,
        )
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        left_sub = message_filters.Subscriber(
            self, Image, str(self.get_parameter("left_image_topic").value),
            qos_profile=qos_profile_sensor_data,
        )
        right_sub = message_filters.Subscriber(
            self, Image, str(self.get_parameter("right_image_topic").value),
            qos_profile=qos_profile_sensor_data,
        )
        self.image_subscribers = (left_sub, right_sub)
        self.synchronizer = message_filters.ApproximateTimeSynchronizer(
            [left_sub, right_sub], queue_size=4,
            slop=float(self.get_parameter("sync_slop").value),
        )
        self.synchronizer.registerCallback(self._on_stereo_pair)

        output_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        self.depth_pub = self.create_publisher(
            Image, str(self.get_parameter("depth_topic").value), output_qos
        )
        self.info_pub = self.create_publisher(
            CameraInfo, str(self.get_parameter("depth_camera_info_topic").value), output_qos
        )
        self.cloud_pub = self.create_publisher(
            PointCloud2, str(self.get_parameter("pointcloud_topic").value), output_qos
        )
        self.worker = threading.Thread(target=self._inference_loop, daemon=True)
        self.worker.start()
        self.get_logger().info("Ready; waiting for rectified infrared images and camera calibration")

    def _load_model(self, model_path: Path) -> torch.nn.Module:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required, but torch.cuda.is_available() is false")
        if not model_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {model_path}")
        cfg_path = model_path.parent / "cfg.yaml"
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Model config not found: {cfg_path}")

        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.set_grad_enabled(False)
        self.get_logger().info(f"Loading checkpoint: {model_path}")
        model = torch.load(model_path, map_location="cpu", weights_only=False)
        model.args.valid_iters = self.valid_iters
        model.args.max_disp = self.max_disp
        model.cuda().eval()
        self.get_logger().info(
            f"Model loaded on {torch.cuda.get_device_name(0)}; "
            f"valid_iters={self.valid_iters}, max_disp={self.max_disp}"
        )
        return model

    def _on_left_info(self, msg: CameraInfo) -> None:
        self.left_info = msg

    def _on_right_info(self, msg: CameraInfo) -> None:
        self.right_info = msg

    def _on_color_info(self, msg: CameraInfo) -> None:
        self.color_info = msg

    def _on_color(self, msg: Image) -> None:
        self.latest_color = msg

    def _on_stereo_pair(self, left: Image, right: Image) -> None:
        if left.width != right.width or left.height != right.height:
            self.get_logger().error("Left/right dimensions differ; dropping stereo pair")
            return
        with self._pair_lock:
            self._latest_pair = (left, right)
            self._frames_received += 1
        self._pair_ready.set()

    @staticmethod
    def _mono_image(msg: Image) -> np.ndarray:
        if msg.encoding.lower() not in ("mono8", "8uc1"):
            raise ValueError(f"Expected mono8 infrared image, received {msg.encoding!r}")
        rows = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
        return np.ascontiguousarray(rows[:, : msg.width])

    def _calibration(self) -> tuple[float, float, CameraInfo] | None:
        left = self.left_info
        right = self.right_info
        if left is None or right is None:
            return None
        fx = float(left.p[0])
        right_fx = float(right.p[0])
        if not math.isfinite(fx) or fx <= 0 or not math.isfinite(right_fx) or right_fx == 0:
            return None
        baseline = abs(float(right.p[3]) / right_fx)
        if not math.isfinite(baseline) or baseline <= 0:
            return None
        return fx, baseline, left

    def _infer(self, left_msg: Image, right_msg: Image) -> np.ndarray:
        left = self._mono_image(left_msg)
        right = self._mono_image(right_msg)
        height, width = left.shape

        # The model was trained for RGB. Repeating Y8 preserves the IR intensity
        # while satisfying the three-channel input contract.
        left_tensor = torch.from_numpy(left).to("cuda", non_blocking=True)
        right_tensor = torch.from_numpy(right).to("cuda", non_blocking=True)
        left_tensor = left_tensor[None, None].expand(-1, 3, -1, -1).float()
        right_tensor = right_tensor[None, None].expand(-1, 3, -1, -1).float()
        padder = InputPadder(left_tensor.shape, divis_by=32, force_square=False)
        left_tensor, right_tensor = padder.pad(left_tensor, right_tensor)

        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=AMP_DTYPE):
            disparity = self.model.forward(
                left_tensor,
                right_tensor,
                iters=self.valid_iters,
                test_mode=True,
                optimize_build_volume=self.backend,
            )
        disparity = padder.unpad(disparity.float())
        return disparity.squeeze().cpu().numpy().reshape(height, width)

    def _publish_depth(
        self, depth: np.ndarray, source: Image, camera_info: CameraInfo
    ) -> None:
        depth = np.ascontiguousarray(depth.astype(np.float32, copy=False))
        msg = Image()
        msg.header = source.header
        msg.height, msg.width = depth.shape
        msg.encoding = "32FC1"
        msg.is_bigendian = False
        msg.step = msg.width * depth.dtype.itemsize
        msg.data = depth.tobytes()
        self.depth_pub.publish(msg)

        info = copy.deepcopy(camera_info)
        info.header = source.header
        self.info_pub.publish(info)

    @staticmethod
    def _stamp_seconds(msg: Image) -> float:
        return float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9

    @staticmethod
    def _color_image(msg: Image) -> np.ndarray:
        encoding = msg.encoding.lower()
        if encoding not in ("rgb8", "bgr8"):
            raise ValueError(f"Expected rgb8 or bgr8 color image, received {msg.encoding!r}")
        rows = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
        image = rows[:, : msg.width * 3].reshape(msg.height, msg.width, 3)
        if encoding == "bgr8":
            image = image[..., ::-1]
        return np.ascontiguousarray(image)

    @staticmethod
    def _quaternion_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
        norm = x * x + y * y + z * z + w * w
        if norm < 1e-12:
            return np.eye(3, dtype=np.float64)
        scale = 2.0 / norm
        return np.array(
            [
                [1 - scale * (y * y + z * z), scale * (x * y - z * w), scale * (x * z + y * w)],
                [scale * (x * y + z * w), 1 - scale * (x * x + z * z), scale * (y * z - x * w)],
                [scale * (x * z - y * w), scale * (y * z + x * w), 1 - scale * (x * x + y * y)],
            ],
            dtype=np.float64,
        )

    def _publish_color_cloud(
        self, depth: np.ndarray, source: Image, left_info: CameraInfo
    ) -> None:
        color_msg = self.latest_color
        color_info = self.color_info
        if color_msg is None or color_info is None:
            return
        if abs(self._stamp_seconds(source) - self._stamp_seconds(color_msg)) > self.max_color_age:
            return

        try:
            transform = self.tf_buffer.lookup_transform(
                color_info.header.frame_id,
                source.header.frame_id,
                rclpy.time.Time(),
            )
        except tf2_ros.TransformException as exc:
            self.get_logger().warning(f"Cannot transform IR points to color camera: {exc}")
            return

        stride = self.cloud_stride
        v, u = np.mgrid[0 : depth.shape[0] : stride, 0 : depth.shape[1] : stride]
        z = depth[::stride, ::stride]
        valid = np.isfinite(z) & (z > 0) & (z <= self.max_depth)
        if not np.any(valid):
            return
        u = u[valid].astype(np.float32)
        v = v[valid].astype(np.float32)
        z = z[valid].astype(np.float32)
        fx, fy, cx, cy = float(left_info.p[0]), float(left_info.p[5]), float(left_info.p[2]), float(left_info.p[6])
        points_left = np.column_stack(((u - cx) * z / fx, (v - cy) * z / fy, z)).astype(np.float32)

        rotation = transform.transform.rotation
        translation = transform.transform.translation
        matrix = self._quaternion_matrix(rotation.x, rotation.y, rotation.z, rotation.w)
        offset = np.array([translation.x, translation.y, translation.z], dtype=np.float64)
        points_color = points_left.astype(np.float64) @ matrix.T + offset

        camera_matrix = np.asarray(color_info.k, dtype=np.float64).reshape(3, 3)
        distortion = np.asarray(color_info.d, dtype=np.float64)
        projected, _ = cv2.projectPoints(
            points_color,
            np.zeros(3),
            np.zeros(3),
            camera_matrix,
            distortion,
        )
        pixels = np.rint(projected.reshape(-1, 2)).astype(np.int32)
        inside = (
            (points_color[:, 2] > 0)
            & (pixels[:, 0] >= 0)
            & (pixels[:, 0] < color_msg.width)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < color_msg.height)
        )
        if not np.any(inside):
            return
        points_left = points_left[inside]
        pixels = pixels[inside]
        rgb_image = self._color_image(color_msg)
        colors = rgb_image[pixels[:, 1], pixels[:, 0]].astype(np.uint32)
        packed_rgb = (colors[:, 0] << 16) | (colors[:, 1] << 8) | colors[:, 2]

        cloud_data = np.empty(
            len(points_left),
            dtype=np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<u4")]),
        )
        cloud_data["x"] = points_left[:, 0]
        cloud_data["y"] = points_left[:, 1]
        cloud_data["z"] = points_left[:, 2]
        cloud_data["rgb"] = packed_rgb

        cloud = PointCloud2()
        cloud.header = source.header
        cloud.height = 1
        cloud.width = len(cloud_data)
        cloud.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.UINT32, count=1),
        ]
        cloud.is_bigendian = False
        cloud.point_step = cloud_data.dtype.itemsize
        cloud.row_step = cloud.point_step * cloud.width
        cloud.data = cloud_data.tobytes()
        cloud.is_dense = True
        self.cloud_pub.publish(cloud)

    def _inference_loop(self) -> None:
        warned_calibration = False
        while not self._stop.is_set():
            if not self._pair_ready.wait(timeout=0.2):
                continue
            self._pair_ready.clear()
            with self._pair_lock:
                pair = self._latest_pair
                self._latest_pair = None
            if pair is None:
                continue

            calibration = self._calibration()
            if calibration is None:
                if not warned_calibration:
                    self.get_logger().warning("Waiting for valid left/right CameraInfo")
                    warned_calibration = True
                continue
            warned_calibration = False
            fx, baseline, camera_info = calibration

            started = time.perf_counter()
            try:
                disparity = self._infer(*pair)
                with np.errstate(divide="ignore", invalid="ignore"):
                    depth = (fx * baseline) / disparity
                columns = np.arange(disparity.shape[1], dtype=np.float32)[None, :]
                invalid = (
                    ~np.isfinite(disparity)
                    | (disparity <= 0)
                    | ((columns - disparity) < 0)
                    | ~np.isfinite(depth)
                    | (depth <= 0)
                    | (depth > self.max_depth)
                )
                depth[invalid] = np.nan
                if self._stop.is_set() or not rclpy.ok():
                    break
                self._publish_depth(depth, pair[0], camera_info)
                self._publish_color_cloud(depth, pair[0], camera_info)
                self._frames_published += 1
                if self._frames_published == 1 or self._frames_published % 30 == 0:
                    elapsed_ms = (time.perf_counter() - started) * 1000.0
                    self.get_logger().info(
                        f"Published frame {self._frames_published}: {elapsed_ms:.1f} ms, "
                        f"baseline={baseline * 1000:.3f} mm, input={pair[0].width}x{pair[0].height}, "
                        f"received={self._frames_received}"
                    )
            except Exception as exc:
                if not self._stop.is_set() and rclpy.ok():
                    self.get_logger().error(f"Inference failed: {exc}")

    def destroy_node(self) -> bool:
        self._stop.set()
        self._pair_ready.set()
        if hasattr(self, "worker"):
            self.worker.join(timeout=10.0)
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node: FastFoundationStereoDepthNode | None = None
    try:
        node = FastFoundationStereoDepthNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
