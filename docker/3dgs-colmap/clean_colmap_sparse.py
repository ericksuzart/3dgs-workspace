#!/usr/bin/env python3
"""
Remove panorama camera folders from COLMAP sparse reconstruction.

Reads sparse/0/ (binary or text format), filters out images matching
given camera prefixes, and writes back the cleaned reconstruction.
"""

import os
import struct
import sys
from collections import namedtuple

# COLMAP format structures

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


def read_intrinsics_text(path):
    cameras = {}
    with open(path, "r") as fid:
        for line in fid:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            camera_id = int(parts[0])
            model_name = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = tuple(map(float, parts[4:]))
            cameras[camera_id] = Camera(
                id=camera_id, model=model_name, width=width, height=height, params=params
            )
    return cameras


def write_intrinsics_text(cameras, path):
    with open(path, "w") as fid:
        for cam in cameras.values():
            params_str = " ".join(f"{p:.6f}" for p in cam.params)
            fid.write(f"{cam.id} {cam.model} {cam.width} {cam.height} {params_str}\n")


def read_extrinsics_text(path):
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Image header line: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
            parts = line.split()
            image_id = int(parts[0])
            qvec = tuple(map(float, parts[1:5]))
            tvec = tuple(map(float, parts[5:8]))
            camera_id = int(parts[8])
            name = parts[9]

            # Points line: N entries of X Y POINT3D_ID
            points_line = fid.readline()
            if not points_line:
                break
            points_parts = points_line.strip().split()
            xys = []
            point3D_ids = []
            for i in range(0, len(points_parts), 3):
                x = float(points_parts[i])
                y = float(points_parts[i + 1])
                p3d_id = int(points_parts[i + 2])
                xys.append((x, y))
                point3D_ids.append(p3d_id)

            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec, camera_id=camera_id,
                name=name, xys=xys, point3D_ids=point3D_ids,
            )
    return images


def write_extrinsics_text(images, path):
    with open(path, "w") as fid:
        for img in images.values():
            qvec = " ".join(f"{v:.9f}" for v in img.qvec)
            tvec = " ".join(f"{v:.9f}" for v in img.tvec)
            fid.write(f"{img.id} {qvec} {tvec} {img.camera_id} {img.name}\n")
            points_str = " ".join(
                f"{x:.6f} {y:.6f} {p3d_id}"
                for (x, y), p3d_id in zip(img.xys, img.point3D_ids)
            )
            fid.write(f"{points_str}\n")


def read_points3d_text(path):
    """Read points3D text format, returning list of parsed lines for reprocessing."""
    lines = []
    with open(path, "r") as fid:
        for line in fid:
            lines.append(line)
    return lines


def filter_points3d_text(lines, removed_image_ids):
    """Filter points3D lines, removing tracks referencing deleted images."""
    out_lines = []
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped or line_stripped.startswith("#"):
            out_lines.append(line)
            continue

        parts = line_stripped.split()
        # POINT3D_ID X Y Z R G B ERROR TRACK[] as IMAGE_ID POINT2D_ID pairs
        # Track starts at index 8
        track_parts = parts[8:]
        filtered_track = []
        for i in range(0, len(track_parts), 2):
            img_id = int(track_parts[i])
            p2d_idx = track_parts[i + 1]
            if img_id not in removed_image_ids:
                filtered_track.append((img_id, p2d_idx))

        if filtered_track:
            header = " ".join(parts[:8])
            track_str = " ".join(f"{img_id} {p2d_idx}" for img_id, p2d_idx in filtered_track)
            out_lines.append(f"{header} {track_str}\n")

    return out_lines


def write_points3d_text(lines, path):
    with open(path, "w") as fid:
        fid.writelines(lines)

def filter_points3d_binary(path, removed_image_ids):
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

            if track_image_ids:
                points.append((point_id, xyz, rgb, error, track_image_ids, track_point2d_idxs))

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

def detect_format(sparse_dir):
    """Detect whether the sparse model uses binary or text format."""
    if os.path.exists(os.path.join(sparse_dir, "images.bin")):
        return "binary"
    if os.path.exists(os.path.join(sparse_dir, "images.txt")):
        return "text"
    return None


def filter_sparse_dir(sparse_dir, remove_prefixes):
    fmt = detect_format(sparse_dir)
    if fmt is None:
        print(f"No COLMAP sparse files found in {sparse_dir}")
        return 0

    is_binary = fmt == "binary"
    images_ext = ".bin" if is_binary else ".txt"
    cameras_ext = ".bin" if is_binary else ".txt"
    points_ext = ".bin" if is_binary else ".txt"

    images_path = os.path.join(sparse_dir, f"images{images_ext}")
    cameras_path = os.path.join(sparse_dir, f"cameras{cameras_ext}")
    points_path = os.path.join(sparse_dir, f"points3D{points_ext}")

    print(f"  Format: {fmt}")

    # Read
    if is_binary:
        images = read_extrinsics_binary(images_path)
        cameras = read_intrinsics_binary(cameras_path)
    else:
        images = read_extrinsics_text(images_path)
        cameras = read_intrinsics_text(cameras_path)

    # Find images to remove (match directory prefix with trailing slash)
    # Generate both naming conventions: pycolmap may store camera names as
    # either pano_cameraN/ or _cameraN/ depending on version.
    to_remove_ids = set()
    search_prefixes = set()
    for p in remove_prefixes:
        base = p + "/" if not p.endswith("/") else p
        search_prefixes.add(base)
        # Also generate the alternate prefix form
        for orig, alt in [("pano_camera", "_camera"), ("_camera", "pano_camera")]:
            if base.startswith(orig):
                search_prefixes.add(alt + base[len(orig):])
    full_prefixes = sorted(search_prefixes)
    for img in images.values():
        if any(img.name.startswith(fp) for fp in full_prefixes):
            to_remove_ids.add(img.id)

    if not to_remove_ids:
        print(f"  No images matching prefixes found in {len(images)} total images.")
        # Debug: show a sample of actual image names to diagnose prefix mismatch
        sample = list(images.values())[:5]
        print(f"  First {len(sample)} image names in sparse model:")
        for img in sample:
            print(f"    \"{img.name}\"")
        print(f"  Searching prefixes: {full_prefixes}")
        return 0

    print(f"  Removing {len(to_remove_ids)} images from sparse reconstruction...")

    kept_images = {iid: img for iid, img in images.items() if iid not in to_remove_ids}
    kept_cameras = cameras  # keep all cameras — intrinsics are shared

    # Write filtered files
    if is_binary:
        write_extrinsics_binary(kept_images, images_path)
        write_intrinsics_binary(kept_cameras, cameras_path)
        if os.path.exists(points_path):
            filter_points3d_binary(points_path, to_remove_ids)
    else:
        write_extrinsics_text(kept_images, images_path)
        write_intrinsics_text(kept_cameras, cameras_path)
        if os.path.exists(points_path):
            points_lines = read_points3d_text(points_path)
            points_lines = filter_points3d_text(points_lines, to_remove_ids)
            write_points3d_text(points_lines, points_path)

    print(f"  Remaining: {len(kept_images)} images, {len(kept_cameras)} cameras")
    return len(to_remove_ids)


def main():
    if len(sys.argv) < 2:
        print("Usage: clean_colmap_sparse.py <dataset_path> [prefix1,prefix2,...]")
        print("Default prefixes: pano_camera0,pano_camera1,pano_camera2,pano_camera3")
        sys.exit(1)

    dataset_path = sys.argv[1]
    if len(sys.argv) >= 3:
        remove_prefixes = sys.argv[2].split(",")
    else:
        remove_prefixes = ["pano_camera0", "pano_camera1", "pano_camera2", "pano_camera3"]

    sparse_dir = os.path.join(dataset_path, "sparse", "0")
    if not os.path.isdir(sparse_dir):
        print(f"ERROR: sparse/0/ not found at {sparse_dir}")
        sys.exit(1)

    print(f"Cleaning COLMAP sparse data: {sparse_dir}")
    print(f"Removing images with prefixes: {remove_prefixes}")

    count = filter_sparse_dir(sparse_dir, remove_prefixes)

    if count > 0:
        print(f"Done! Removed {count} images from sparse reconstruction.")
    else:
        print("No changes needed.")


if __name__ == "__main__":
    main()
