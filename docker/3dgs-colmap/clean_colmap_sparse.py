#!/usr/bin/env python3
"""
Remove panorama camera folders from COLMAP sparse reconstruction.

Reads sparse/0/ binary files, filters out images from specified camera prefixes,
and writes back the cleaned reconstruction.
"""

import os
import struct
import sys
from collections import namedtuple

# COLMAP Binary Format

Camera = namedtuple("Camera", ["id", "model", "width", "height", "params"])
Image = namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])

CAMERA_MODEL_IDS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}
CAMERA_MODEL_NAMES = {v[0]: k for k, v in CAMERA_MODEL_IDS.items()}


def read_intrinsics_binary(path):
    cameras = {}
    with open(path, "rb") as fid:
        num_cameras = struct.unpack("<Q", fid.read(8))[0]
        for _ in range(num_cameras):
            camera_id, model_id = struct.unpack("<ii", fid.read(8))
            width, height = struct.unpack("<QQ", fid.read(16))
            model_name, num_params = CAMERA_MODEL_IDS[model_id]
            params = struct.unpack("<" + "d" * num_params, fid.read(8 * num_params))
            cameras[camera_id] = Camera(
                id=camera_id, model=model_name, width=width, height=height, params=params
            )
    return cameras


def write_intrinsics_binary(cameras, path):
    with open(path, "wb") as fid:
        fid.write(struct.pack("<Q", len(cameras)))
        for cam in cameras.values():
            model_id = CAMERA_MODEL_NAMES[cam.model]
            fid.write(struct.pack("<ii", cam.id, model_id))
            fid.write(struct.pack("<QQ", cam.width, cam.height))
            fid.write(struct.pack("<" + "d" * len(cam.params), *cam.params))


def read_extrinsics_binary(path):
    images = {}
    with open(path, "rb") as fid:
        num_images = struct.unpack("<Q", fid.read(8))[0]
        for _ in range(num_images):
            image_id = struct.unpack("<Q", fid.read(8))[0]
            qvec = struct.unpack("<dddd", fid.read(32))
            tvec = struct.unpack("<ddd", fid.read(24))
            camera_id = struct.unpack("<i", fid.read(4))[0]

            # Null-terminated image name
            name_bytes = b""
            while True:
                c = fid.read(1)
                if c == b"\x00":
                    break
                name_bytes += c
            name = name_bytes.decode("utf-8")

            num_points2d = struct.unpack("<Q", fid.read(8))[0]
            xys = []
            point3D_ids = []
            for _ in range(num_points2d):
                x, y = struct.unpack("<dd", fid.read(16))
                point3D_id = struct.unpack("<q", fid.read(8))[0]
                xys.append((x, y))
                point3D_ids.append(point3D_id)

            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec, camera_id=camera_id,
                name=name, xys=xys, point3D_ids=point3D_ids,
            )
    return images


def write_extrinsics_binary(images, path):
    with open(path, "wb") as fid:
        fid.write(struct.pack("<Q", len(images)))
        for img in images.values():
            fid.write(struct.pack("<Q", img.id))
            fid.write(struct.pack("<dddd", *img.qvec))
            fid.write(struct.pack("<ddd", *img.tvec))
            fid.write(struct.pack("<i", img.camera_id))
            fid.write(img.name.encode("utf-8") + b"\x00")
            fid.write(struct.pack("<Q", len(img.xys)))
            for (x, y), p3d in zip(img.xys, img.point3D_ids):
                fid.write(struct.pack("<ddq", x, y, p3d))


def filter_sparse_dir(sparse_dir, remove_prefixes):
    images_bin = os.path.join(sparse_dir, "images.bin")
    cameras_bin = os.path.join(sparse_dir, "cameras.bin")
    points3d_bin = os.path.join(sparse_dir, "points3D.bin")

    if not os.path.exists(images_bin):
        print(f"No images.bin found in {sparse_dir}")
        return 0

    images = read_extrinsics_binary(images_bin)
    cameras = read_intrinsics_binary(cameras_bin)

    # Find images to remove (match full directory prefix with trailing slash)
    to_remove_ids = set()
    full_prefixes = [p + "/" if not p.endswith("/") else p for p in remove_prefixes]
    for img in images.values():
        if any(img.name.startswith(fp) for fp in full_prefixes):
            to_remove_ids.add(img.id)

    if not to_remove_ids:
        print("No images matching prefixes found. Nothing to remove.")
        return 0

    print(f"Removing {len(to_remove_ids)} images from sparse reconstruction...")

    # Keep only wanted images
    kept_images = {iid: img for iid, img in images.items() if iid not in to_remove_ids}

    # Keep all cameras — camera intrinsics are shared and still valid
    kept_cameras = cameras

    # Write filtered images and cameras
    write_extrinsics_binary(kept_images, images_bin)
    write_intrinsics_binary(kept_cameras, cameras_bin)

    # Filter points3D.bin — remove track entries for removed images, drop orphaned points
    if os.path.exists(points3d_bin):
        filter_points3d(points3d_bin, to_remove_ids)

    print(f"Remaining: {len(kept_images)} images, {len(kept_cameras)} cameras")
    return len(to_remove_ids)


def filter_points3d(path, removed_image_ids):
    """Read points3D.bin, filter out track entries for removed images, write back."""
    points = []
    with open(path, "rb") as fid:
        num_points = struct.unpack("<Q", fid.read(8))[0]
        for _ in range(num_points):
            point_id = struct.unpack("<Q", fid.read(8))[0]
            xyz = struct.unpack("<ddd", fid.read(24))
            rgb = struct.unpack("<BBB", fid.read(3))
            error = struct.unpack("<d", fid.read(8))[0]
            track_length = struct.unpack("<Q", fid.read(8))[0]

            track_image_ids = []
            track_point2d_idxs = []
            for _ in range(track_length):
                img_id, p2d_idx = struct.unpack("<ii", fid.read(8))
                if img_id not in removed_image_ids:
                    track_image_ids.append(img_id)
                    track_point2d_idxs.append(p2d_idx)

            # Only keep points still observed by at least one remaining image
            if track_image_ids:
                points.append((point_id, xyz, rgb, error, track_image_ids, track_point2d_idxs))

    # Write back
    with open(path, "wb") as fid:
        fid.write(struct.pack("<Q", len(points)))
        for point_id, xyz, rgb, error, track_image_ids, track_point2d_idxs in points:
            fid.write(struct.pack("<Q", point_id))
            fid.write(struct.pack("<ddd", *xyz))
            fid.write(struct.pack("<BBB", *rgb))
            fid.write(struct.pack("<d", error))
            fid.write(struct.pack("<Q", len(track_image_ids)))
            for img_id, p2d_idx in zip(track_image_ids, track_point2d_idxs):
                fid.write(struct.pack("<ii", img_id, p2d_idx))

    removed = num_points - len(points)
    print(f"Points3D: {num_points} -> {len(points)} (removed {removed} orphaned points)")


def main():
    if len(sys.argv) < 2:
        print("Usage: clean_colmap_sparse.py <dataset_path> [prefix1,prefix2,...]")
        print("Default prefixes: _camera0,_camera1,_camera2,_camera3")
        sys.exit(1)

    dataset_path = sys.argv[1]
    if len(sys.argv) >= 3:
        remove_prefixes = sys.argv[2].split(",")
    else:
        remove_prefixes = ["_camera0", "_camera1", "_camera2", "_camera3"]

    sparse_dir = os.path.join(dataset_path, "sparse", "0")
    if not os.path.isdir(sparse_dir):
        print(f"ERROR: sparse/0/ not found at {sparse_dir}")
        sys.exit(1)

    print(f"Cleaning COLMAP sparse data in: {sparse_dir}")
    print(f"Removing images with prefixes: {remove_prefixes}")
    print()

    count = filter_sparse_dir(sparse_dir, remove_prefixes)

    if count > 0:
        print(f"\nDone! Removed {count} images from sparse reconstruction.")
    else:
        print("No changes needed.")


if __name__ == "__main__":
    main()
