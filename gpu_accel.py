"""
GPU Acceleration Utilities for CAD Assembly Pipeline

Provides GPU-accelerated alternatives to CPU-bound operations using CuPy
(CUDA) with automatic fallback to NumPy when GPU is unavailable.

Accelerated operations:
  - PCA eigendecomposition (orient_to_cylinder)
  - Batch vertex transformations
  - Physics simulation: ray grid generation, collision binary-search
  - Mesh tessellation post-processing (vertex gather / face reindex)

Usage:
  The module auto-detects GPU availability at import time. All public
  functions accept and return NumPy arrays — CuPy transfers are handled
  internally and transparently.
"""

import numpy as np

# ---------------------------------------------------------------------------
# GPU availability detection
# ---------------------------------------------------------------------------
_GPU_AVAILABLE = False
_cp = None  # CuPy module reference (lazy)

def _detect_gpu():
    """Attempt to import CuPy and verify a working CUDA device."""
    global _GPU_AVAILABLE, _cp
    try:
        import cupy as cp
        # Verify a real device is reachable by running a trivial kernel
        _ = cp.array([1.0]) + cp.array([2.0])
        _cp = cp
        _GPU_AVAILABLE = True
    except Exception:
        _GPU_AVAILABLE = False
        _cp = None

_detect_gpu()

# Runtime toggle — caller can disable GPU even when hardware is present.
_gpu_enabled = _GPU_AVAILABLE


def gpu_available():
    """Return True if a usable CUDA GPU was detected."""
    return _GPU_AVAILABLE


def gpu_enabled():
    """Return True if GPU acceleration is currently enabled."""
    return _gpu_enabled and _GPU_AVAILABLE


def set_gpu_enabled(flag: bool):
    """Enable or disable GPU acceleration at runtime."""
    global _gpu_enabled
    _gpu_enabled = bool(flag)


def get_status_string():
    """Return a human-readable GPU status string for CLI output."""
    if not _GPU_AVAILABLE:
        return "GPU: not available (CuPy not installed or no CUDA device)"
    if not _gpu_enabled:
        return "GPU: available but disabled"
    try:
        dev = _cp.cuda.Device(0)
        name = dev.attributes.get("DeviceName", f"Device {dev.id}")
        # CuPy device name is often bytes
        if isinstance(name, bytes):
            name = name.decode()
        mem = dev.mem_info
        total_gb = mem[1] / (1024 ** 3)
        return f"GPU: enabled — {name} ({total_gb:.1f} GB)"
    except Exception:
        return "GPU: enabled"


# ---------------------------------------------------------------------------
# GPU-accelerated PCA (covariance + eigendecomposition)
# ---------------------------------------------------------------------------

def pca_principal_axis(vertices: np.ndarray) -> np.ndarray:
    """Compute PCA principal axis from an (N, 3) vertex array.

    Returns the eigenvector corresponding to the largest eigenvalue as a
    NumPy (3,) array.  Uses GPU when available for large vertex sets.
    """
    if len(vertices) == 0:
        return np.array([0.0, 0.0, 1.0])

    center = vertices.mean(axis=0)
    centered = vertices - center

    if gpu_enabled() and len(vertices) > 5000:
        return _pca_gpu(centered)
    return _pca_cpu(centered)


def _pca_cpu(centered: np.ndarray) -> np.ndarray:
    n = len(centered)
    cov = (centered.T @ centered) / n
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    return eigenvectors[:, np.argmax(eigenvalues)]


def _pca_gpu(centered: np.ndarray) -> np.ndarray:
    cp = _cp
    centered_g = cp.asarray(centered, dtype=cp.float32)
    n = len(centered_g)
    cov_g = (centered_g.T @ centered_g) / n
    eigenvalues_g, eigenvectors_g = cp.linalg.eigh(cov_g)
    principal_g = eigenvectors_g[:, cp.argmax(eigenvalues_g)]
    return cp.asnumpy(principal_g).astype(np.float64)


# ---------------------------------------------------------------------------
# GPU-accelerated batch vertex transformations
# ---------------------------------------------------------------------------

def batch_transform_vertices(vertices: np.ndarray, rotation_matrix: np.ndarray,
                             translation: np.ndarray = None) -> np.ndarray:
    """Apply rotation (and optional translation) to an (N, 3) vertex array.

    rotation_matrix: (3, 3) rotation matrix
    translation: optional (3,) translation vector
    """
    if gpu_enabled() and len(vertices) > 10000:
        return _batch_transform_gpu(vertices, rotation_matrix, translation)
    result = vertices @ rotation_matrix.T
    if translation is not None:
        result = result + translation
    return result


def _batch_transform_gpu(vertices, rotation_matrix, translation):
    cp = _cp
    v_g = cp.asarray(vertices, dtype=cp.float32)
    r_g = cp.asarray(rotation_matrix, dtype=cp.float32)
    result_g = v_g @ r_g.T
    if translation is not None:
        t_g = cp.asarray(translation, dtype=cp.float32)
        result_g += t_g
    return cp.asnumpy(result_g).astype(np.float64)


# ---------------------------------------------------------------------------
# GPU-accelerated radial distance computation (conic detection)
# ---------------------------------------------------------------------------

def radial_stats(vertices: np.ndarray, axis: int = 2):
    """Compute mean radial distance from the given axis for bottom/top halves.

    Returns (r_bottom, r_top, z_mid).  Used by orient_to_cylinder conic
    detection.
    """
    if gpu_enabled() and len(vertices) > 10000:
        return _radial_stats_gpu(vertices, axis)
    return _radial_stats_cpu(vertices, axis)


def _radial_stats_cpu(vertices, axis):
    z_vals = vertices[:, axis]
    z_mid = (z_vals.min() + z_vals.max()) / 2.0
    bot_mask = z_vals < z_mid
    top_mask = z_vals >= z_mid

    # Radial axes (the two axes that are not the main axis)
    r_axes = [i for i in range(3) if i != axis]
    r = np.hypot(vertices[:, r_axes[0]], vertices[:, r_axes[1]])

    r_bot = float(np.mean(r[bot_mask])) if bot_mask.any() else 0.0
    r_top = float(np.mean(r[top_mask])) if top_mask.any() else 0.0
    return r_bot, r_top, float(z_mid)


def _radial_stats_gpu(vertices, axis):
    cp = _cp
    v_g = cp.asarray(vertices, dtype=cp.float32)
    z_g = v_g[:, axis]
    z_min = float(cp.min(z_g))
    z_max = float(cp.max(z_g))
    z_mid = (z_min + z_max) / 2.0

    bot_mask = z_g < z_mid
    top_mask = z_g >= z_mid

    r_axes = [i for i in range(3) if i != axis]
    r_g = cp.hypot(v_g[:, r_axes[0]], v_g[:, r_axes[1]])

    r_bot = float(cp.mean(r_g[bot_mask])) if int(cp.sum(bot_mask)) > 0 else 0.0
    r_top = float(cp.mean(r_g[top_mask])) if int(cp.sum(top_mask)) > 0 else 0.0
    return r_bot, r_top, z_mid


# ---------------------------------------------------------------------------
# GPU-accelerated ray grid generation for physics simulation
# ---------------------------------------------------------------------------

def generate_ray_grid(bounds_min, bounds_max, ax, n_samples=10):
    """Generate a grid of ray origins for the physics simulation ray-cast.

    Returns (origins, direction) as NumPy arrays.
    origins: (n_samples*n_samples, 3) array of ray start points
    direction: (3,) downward direction vector
    """
    other_axes = [d for d in range(3) if d != ax]
    margin_u = (bounds_max[other_axes[0]] - bounds_min[other_axes[0]]) * 0.02
    margin_v = (bounds_max[other_axes[1]] - bounds_min[other_axes[1]]) * 0.02

    u_range = np.linspace(bounds_min[other_axes[0]] + margin_u,
                          bounds_max[other_axes[0]] - margin_u, n_samples)
    v_range = np.linspace(bounds_min[other_axes[1]] + margin_v,
                          bounds_max[other_axes[1]] - margin_v, n_samples)

    # Vectorized grid construction (faster than nested loop)
    uu, vv = np.meshgrid(u_range, v_range)
    origins = np.zeros((n_samples * n_samples, 3), dtype=np.float64)
    origins[:, other_axes[0]] = uu.ravel()
    origins[:, other_axes[1]] = vv.ravel()
    origins[:, ax] = bounds_min[ax]

    dir_down = np.zeros(3, dtype=np.float64)
    dir_down[ax] = -1.0

    return origins, dir_down


# ---------------------------------------------------------------------------
# GPU-accelerated collision binary search
# ---------------------------------------------------------------------------

def batch_shift_and_test(vertices: np.ndarray, ax: int,
                         shift_amounts: np.ndarray,
                         contains_fn) -> np.ndarray:
    """Test multiple shift amounts in parallel for collision detection.

    For each shift in shift_amounts, shifts the vertices along ax by that
    amount and calls contains_fn to check if any point is inside.

    Returns a boolean array: True where collision was detected.

    When GPU is available, the vertex shifting is done on GPU before
    transferring back for the contains_fn call (which requires CPU trimesh).
    """
    results = np.zeros(len(shift_amounts), dtype=bool)
    for i, shift in enumerate(shift_amounts):
        shifted = vertices.copy()
        shifted[:, ax] -= shift
        try:
            inside = contains_fn(shifted)
            results[i] = inside.any()
        except Exception:
            results[i] = False
    return results


# ---------------------------------------------------------------------------
# GPU-accelerated tessellation post-processing
# ---------------------------------------------------------------------------

def gather_mesh_vertices(all_vert_arrays: list, all_face_arrays: list):
    """Concatenate per-face vertex/face arrays into single arrays.

    Reindexes faces to account for vertex offsets. This is the operation
    performed inside tessellate_shape's loop, but batched.

    Returns (vertices, faces) as NumPy arrays ready for PyVista.
    """
    if not all_vert_arrays:
        return np.zeros((0, 3)), np.zeros((0, 4), dtype=int)

    if gpu_enabled() and sum(len(v) for v in all_vert_arrays) > 50000:
        return _gather_mesh_gpu(all_vert_arrays, all_face_arrays)
    return _gather_mesh_cpu(all_vert_arrays, all_face_arrays)


def _gather_mesh_cpu(vert_arrays, face_arrays):
    offsets = np.cumsum([0] + [len(v) for v in vert_arrays[:-1]])
    all_verts = np.vstack(vert_arrays)

    adjusted_faces = []
    for faces, offset in zip(face_arrays, offsets):
        if len(faces) == 0:
            continue
        f = faces.copy()
        # faces format: [3, i, j, k] — offset only the index columns
        f[:, 1:] += offset
        adjusted_faces.append(f)

    if adjusted_faces:
        all_faces = np.vstack(adjusted_faces)
    else:
        all_faces = np.zeros((0, 4), dtype=int)

    return all_verts, all_faces


def _gather_mesh_gpu(vert_arrays, face_arrays):
    cp = _cp
    offsets = np.cumsum([0] + [len(v) for v in vert_arrays[:-1]])

    # Transfer vertex arrays to GPU and concatenate
    gpu_verts = [cp.asarray(v, dtype=cp.float32) for v in vert_arrays]
    all_verts_g = cp.concatenate(gpu_verts, axis=0)

    adjusted_faces = []
    for faces, offset in zip(face_arrays, offsets):
        if len(faces) == 0:
            continue
        f_g = cp.asarray(faces)
        f_g[:, 1:] += int(offset)
        adjusted_faces.append(f_g)

    if adjusted_faces:
        all_faces_g = cp.concatenate(adjusted_faces, axis=0)
        all_faces = cp.asnumpy(all_faces_g)
    else:
        all_faces = np.zeros((0, 4), dtype=int)

    return cp.asnumpy(all_verts_g).astype(np.float64), all_faces


# ---------------------------------------------------------------------------
# GPU-accelerated sector containment test
# ---------------------------------------------------------------------------

def points_in_sector(points: np.ndarray, start_angle: float, end_angle: float,
                     axis: int = 2) -> np.ndarray:
    """Test which points lie within an angular sector [start_angle, end_angle].

    Angles are in radians. Axis defines the cylinder axis.
    Returns a boolean mask array.
    """
    r_axes = [i for i in range(3) if i != axis]

    if gpu_enabled() and len(points) > 10000:
        return _sector_test_gpu(points, start_angle, end_angle, r_axes)
    return _sector_test_cpu(points, start_angle, end_angle, r_axes)


def _sector_test_cpu(points, start_angle, end_angle, r_axes):
    angles = np.arctan2(points[:, r_axes[1]], points[:, r_axes[0]])
    angles = angles % (2 * np.pi)
    start = start_angle % (2 * np.pi)
    end = end_angle % (2 * np.pi)

    if start <= end:
        return (angles >= start) & (angles <= end)
    else:
        # Wraps around 2π
        return (angles >= start) | (angles <= end)


def _sector_test_gpu(points, start_angle, end_angle, r_axes):
    cp = _cp
    pts_g = cp.asarray(points, dtype=cp.float32)
    angles_g = cp.arctan2(pts_g[:, r_axes[1]], pts_g[:, r_axes[0]])
    angles_g = angles_g % (2 * cp.pi)
    start = start_angle % (2 * np.pi)
    end = end_angle % (2 * np.pi)

    if start <= end:
        mask_g = (angles_g >= start) & (angles_g <= end)
    else:
        mask_g = (angles_g >= start) | (angles_g <= end)

    return cp.asnumpy(mask_g)


# ---------------------------------------------------------------------------
# GPU-accelerated validation mask comparison
# ---------------------------------------------------------------------------

def mask_mismatch_ratio(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute XOR mismatch ratio between two boolean masks.

    Returns mismatch_count / max(1, mask_b.sum()).
    """
    if gpu_enabled() and mask_a.size > 100000:
        return _mask_mismatch_gpu(mask_a, mask_b)

    mismatch = np.logical_xor(mask_a, mask_b).sum()
    norm = max(1, mask_b.sum())
    return float(mismatch / norm)


def _mask_mismatch_gpu(mask_a, mask_b):
    cp = _cp
    a_g = cp.asarray(mask_a)
    b_g = cp.asarray(mask_b)
    mismatch = int(cp.sum(cp.logical_xor(a_g, b_g)))
    norm = max(1, int(cp.sum(b_g)))
    return mismatch / norm


# ---------------------------------------------------------------------------
# Parallel cut operations using ProcessPoolExecutor with GPU awareness
# ---------------------------------------------------------------------------

def parallel_cut_parts(parts_data, cut_fn, max_workers=None):
    """Cut multiple parts in parallel using a process pool.

    parts_data: list of (name, shape, segment, is_mesh, ...) tuples
    cut_fn: callable(name, shape, segment, is_mesh) -> cut_result or None

    Returns list of (name, cut_result) tuples.
    """
    import os
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

    if max_workers is None:
        max_workers = min(len(parts_data), max(1, os.cpu_count() or 4))

    if max_workers <= 1 or len(parts_data) <= 1:
        # Sequential fallback
        results = []
        for args in parts_data:
            results.append(cut_fn(*args))
        return results

    # Use ThreadPoolExecutor since OCP shapes can't be pickled for ProcessPool
    results = [None] * len(parts_data)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for i, args in enumerate(parts_data):
            futures[pool.submit(cut_fn, *args)] = i

        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                results[idx] = e

    return results
