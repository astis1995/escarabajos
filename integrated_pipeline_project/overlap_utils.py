import numpy as np
import cv2
import statistics
import time
import os
import re
from collections import defaultdict
import concurrent.futures
from collections import Counter
import unittest

def load_grayscale_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {path}")
    return img

def compute_overlap_mse(small_img, big_img, min_overlap=1, max_overlap=None):
    """
    Computes the best horizontal overlap between two grayscale images by minimizing MSE.
    
    Parameters:
    - small_img (np.ndarray): left image, 2D grayscale
    - big_img (np.ndarray): right image, 2D grayscale
    - min_overlap (int): minimum number of columns to consider as overlap
    - max_overlap (int or None): maximum number of columns to consider (defaults to image width)
    
    Returns:
    - best_overlap (int): number of columns with lowest MSE
    - best_mse (float): corresponding mean squared error
    """
    if small_img.shape != big_img.shape:
        raise ValueError("Images must have the same shape.")

    if small_img.ndim != 2 or big_img.ndim != 2:
        raise ValueError("Images must be 2D grayscale arrays.")

    height, width = small_img.shape
    max_overlap = max_overlap or width

    best_overlap = None
    best_mse = float('inf')

    for overlap in range(min_overlap, max_overlap):
        if overlap <= 0 or overlap > width:
            continue

        small_part = small_img[:, -overlap:]
        big_part = big_img[:, :overlap]

        if small_part.shape != big_part.shape or small_part.size == 0:
            continue  # skip invalid comparisons

        mse = np.mean((small_part.astype(float) - big_part.astype(float)) ** 2)

        if mse < best_mse:
            best_mse = mse
            best_overlap = overlap

    if best_overlap is None:
        raise RuntimeError("No valid overlap found between the images.")

    return best_overlap, best_mse
    


def image_pairs_by_v(folder_path):
    """
    Generator that yields pairs of consecutive images with the same 'v' value.
    
    Filename format: 'v-h_*.tif', e.g., '511-01_avg.tif'
    """
    pattern = re.compile(r"(\d+)-(\d+)_.*\.tif$", re.IGNORECASE)
    groups = defaultdict(list)

    # List and group files by 'v'
    for fname in os.listdir(folder_path):
        match = pattern.match(fname)
        if match:
            v = match.group(1)
            h = int(match.group(2))  # ensure numeric sort
            groups[v].append((h, fname))

    # Yield pairs within each group sorted by h
    for v, items in groups.items():
        sorted_items = sorted(items)  # sorted by h
        for i in range(len(sorted_items) - 1):
            yield os.path.join(folder_path, sorted_items[i][1]), os.path.join(folder_path, sorted_items[i + 1][1])
            

def compute_overlap_from_generator(image_pair_generator, verbose=False):
    overlaps = []
    mses = []
    total_pairs = 0
    valid_pairs = 0
    skipped_pairs = 0
    start_time = time.time()

    for path_small, path_big in image_pair_generator:
        total_pairs += 1
        try:
            small = load_grayscale_image(path_small)
            big = load_grayscale_image(path_big)

            best_overlap, best_mse = compute_overlap_mse(small, big)

            if best_overlap is not None and best_mse is not None:
                overlaps.append(best_overlap)
                mses.append(best_mse)
                valid_pairs += 1

                small_part = small[:, -best_overlap:]
                big_part = big[:, :best_overlap]
                pb1, pb2, pb_both = compute_black_overlap_stats(small_part, big_part)

                if verbose:
                    print(f" Processed: {os.path.basename(path_small)} ↔ {os.path.basename(path_big)} | "
                          f"Overlap: {best_overlap}, MSE: {best_mse:.2f} | "
                          f"Black %: small={pb1:.1f}%, big={pb2:.1f}%, both={pb_both:.1f}%")

            else:
                skipped_pairs += 1
                if verbose:
                    print(f"⚠ Skipped: {path_small} ↔ {path_big} | No valid result")

        except Exception as e:
            skipped_pairs += 1
            if verbose:
                print(f" Error: {path_small} ↔ {path_big} | {str(e)}")

    if not overlaps:
        raise RuntimeError("No valid image pairs processed.")

    median_overlap = statistics.median(overlaps)
    median_mse = statistics.median(mses)
    elapsed = time.time() - start_time

    if verbose:
        print("\n Processing Summary:")
        print(f"   Total pairs:   {total_pairs}")
        print(f"   Valid pairs:   {valid_pairs}")
        print(f"   Skipped pairs: {skipped_pairs}")
        print(f"   Elapsed time:  {elapsed:.2f}s")
        print(f"   Median Overlap: {median_overlap}, Median MSE: {median_mse:.2f}")

    return median_overlap, median_mse
    
def safe_process_with_fn(pair, process_fn):
    try:
        result = process_fn(pair)
        return (pair, result)
    except Exception as e:
        return (pair, None, str(e))

def process_pair_cpu(pair):
    path_small, path_big = pair
    small = load_grayscale_image(path_small)
    big = load_grayscale_image(path_big)
    return compute_overlap_mse(small, big)

def process_pair_gpu(pair):
    path_small, path_big = pair
    small = load_grayscale_image(path_small)
    big = load_grayscale_image(path_big)
    return compute_all_overlaps_gpu(small, big)


       
def compute_overlap_parallel(image_pair_generator, max_workers=None, verbose=False, max_black_pct=92.0):
    overlaps = []
    mses = []
    total_pairs = 0
    valid_pairs = 0
    skipped_pairs = 0
    start_time = time.time()

    image_pairs = list(image_pair_generator)
    total_pairs = len(image_pairs)

    log_lines = []

    from functools import partial

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        safe = partial(safe_process_with_fn, process_fn=process_pair_cpu)

        results = executor.map(safe, image_pairs)


        for result in results:
            if len(result) == 2:
                (path_small, path_big), (best_overlap, best_mse) = result

                # Load again to get overlap slices
                small = load_grayscale_image(path_small)
                big = load_grayscale_image(path_big)
                small_part = small[:, -best_overlap:]
                big_part = big[:, :best_overlap]
                pb1, pb2, pb_both = compute_black_overlap_stats(small_part, big_part)

                if pb1 > max_black_pct or pb2 > max_black_pct:
                    skipped_pairs += 1
                    line = (f" Skipped: {os.path.basename(path_small)} ↔ {os.path.basename(path_big)} | "
                            f"Black % too high: small={pb1:.1f}%, big={pb2:.1f}%")
                    if verbose:
                        print(line)
                    log_lines.append(line)
                    continue

                overlaps.append(best_overlap)
                mses.append(best_mse)
                valid_pairs += 1

                line = (f" Processed: {os.path.basename(path_small)} ↔ {os.path.basename(path_big)} | "
                        f"Overlap: {best_overlap}, MSE: {best_mse:.2f} | "
                        f"Black %: small={pb1:.1f}%, big={pb2:.1f}%, both={pb_both:.1f}%")
                if verbose:
                    print(line)
                log_lines.append(line)

            else:
                (path_small, path_big), _, error_msg = result
                skipped_pairs += 1
                line = f" Error: {os.path.basename(path_small)} ↔ {os.path.basename(path_big)} | {error_msg}"
                if verbose:
                    print(line)
                log_lines.append(line)

    if not overlaps:
        raise RuntimeError("No valid image pairs processed.")

    mode_overlap = Counter(overlaps).most_common(1)[0][0]
    median_mse = statistics.median(mses)
    elapsed = time.time() - start_time

    summary = [
        "\n Parallel Processing Summary:",
        f"   Total pairs:   {total_pairs}",
        f"   Valid pairs:   {valid_pairs}",
        f"   Skipped pairs: {skipped_pairs}",
        f"   Elapsed time:  {elapsed:.2f}s",
        f"   Mode Overlap:  {mode_overlap}, Median MSE: {median_mse:.2f}"
    ]

    for line in summary:
        if verbose:
            print(line)
        log_lines.append(line)

    # Save all logs to file
    with open("stats.txt", "w", encoding="utf-8") as f:

        for line in log_lines:
            f.write(line + "\n")

    return mode_overlap, median_mse


import cupy as cp
import numpy as np
from skimage.io import imread


def compute_overlap_gpu(image_pair_generator, verbose=False, max_black_pct=92.0):
    overlaps = []
    mses = []
    total_pairs = 0
    valid_pairs = 0
    skipped_pairs = 0
    start_time = time.time()

    image_pairs = list(image_pair_generator)
    total_pairs = len(image_pairs)

    log_lines = []
    from functools import partial
    safe = partial(safe_process_with_fn, process_fn=process_pair_gpu)

    with concurrent.futures.ThreadPoolExecutor() as executor:  # no multiprocessing on GPU
        results = executor.map(safe, image_pairs)

        for result in results:
            if len(result) == 2:
                (path_small, path_big), (best_overlap, best_mse) = result

                small = load_grayscale_image(path_small)
                big = load_grayscale_image(path_big)
                small_part = small[:, -best_overlap:]
                big_part = big[:, :best_overlap]
                pb1, pb2, pb_both = compute_black_overlap_stats(small_part, big_part)

                if pb1 > max_black_pct or pb2 > max_black_pct:
                    skipped_pairs += 1
                    line = (f"⚠️ Skipped: {os.path.basename(path_small)} ↔ {os.path.basename(path_big)} | "
                            f"Black % too high: small={pb1:.1f}%, big={pb2:.1f}%")
                    if verbose: print(line)
                    log_lines.append(line)
                    continue

                overlaps.append(best_overlap)
                mses.append(best_mse)
                valid_pairs += 1
                line = (f" Processed: {os.path.basename(path_small)} ↔ {os.path.basename(path_big)} | "
                        f"Overlap: {best_overlap}, MSE: {best_mse:.2f} | "
                        f"Black %: small={pb1:.1f}%, big={pb2:.1f}%, both={pb_both:.1f}%")
                if verbose: print(line)
                log_lines.append(line)
            else:
                (path_small, path_big), _, error_msg = result
                skipped_pairs += 1
                line = f" Error: {os.path.basename(path_small)} ↔ {os.path.basename(path_big)} | {error_msg}"
                if verbose: print(line)
                log_lines.append(line)

    if not overlaps:
        raise RuntimeError("No valid image pairs processed.")

    mode_overlap = Counter(overlaps).most_common(1)[0][0]
    median_mse = statistics.median(mses)
    elapsed = time.time() - start_time

    summary = [
        "\n GPU Processing Summary:",
        f"   Total pairs:   {total_pairs}",
        f"   Valid pairs:   {valid_pairs}",
        f"   Skipped pairs: {skipped_pairs}",
        f"   Elapsed time:  {elapsed:.2f}s",
        f"   Mode Overlap:  {mode_overlap}, Median MSE: {median_mse:.2f}"
    ]
    for line in summary:
        if verbose: print(line)
        log_lines.append(line)

    with open("stats.txt", "w", encoding="utf-8") as f:
        for line in log_lines:
            f.write(line + "\n")

    return mode_overlap, median_mse



def process_pair_gpu_optimized(pair):
    path_small, path_big = pair
    small = load_grayscale_image(path_small)
    big = load_grayscale_image(path_big)

    # Ensure contiguous memory for GPU transfer
    small = np.ascontiguousarray(small, dtype=np.uint8)
    big = np.ascontiguousarray(big, dtype=np.uint8)

    # Transfer to GPU
    small_cp = cp.asarray(small, dtype=cp.float32)
    big_cp = cp.asarray(big, dtype=cp.float32)
    h = small_cp.shape[1]
    mses = cp.zeros(h, dtype=cp.float32)

    for offset in range(1, h + 1):
        s = small_cp[:, -offset:]
        b = big_cp[:, :offset]
        mses[offset - 1] = cp.mean((s - b) ** 2)

    best_overlap = int(cp.argmin(mses).item())
    best_mse = float(mses[best_overlap].item())
    best_overlap += 1

    # Return sliced parts from CPU arrays (no need to re-load)
    small_part = small[:, -best_overlap:]
    big_part = big[:, :best_overlap]

    return best_overlap, best_mse, small_part, big_part


def compute_overlap_gpu_optimized(image_pair_generator, verbose=False, max_black_pct=92.0):
    overlaps = []
    mses = []
    total_pairs = 0
    valid_pairs = 0
    skipped_pairs = 0
    start_time = time.time()

    image_pairs = list(image_pair_generator)
    total_pairs = len(image_pairs)

    log_lines = []

    for pair in image_pairs:
        try:
            best_overlap, best_mse, small_part, big_part = process_pair_gpu_optimized(pair)
            pb1, pb2, pb_both = compute_black_overlap_stats(small_part, big_part)

            if pb1 > max_black_pct or pb2 > max_black_pct:
                skipped_pairs += 1
                line = (f"Skipped: {os.path.basename(pair[0])} ↔ {os.path.basename(pair[1])} | "
                        f"Black % too high: small={pb1:.1f}%, big={pb2:.1f}%")
                if verbose:
                    print(line)
                log_lines.append(line)
                continue

            overlaps.append(best_overlap)
            mses.append(best_mse)
            valid_pairs += 1
            line = (f" Processed: {os.path.basename(pair[0])} ↔ {os.path.basename(pair[1])} | "
                    f"Overlap: {best_overlap}, MSE: {best_mse:.2f} | "
                    f"Black %: small={pb1:.1f}%, big={pb2:.1f}%, both={pb_both:.1f}%")
            if verbose:
                print(line)
            log_lines.append(line)

        except Exception as e:
            skipped_pairs += 1
            line = f" Error: {os.path.basename(pair[0])} ↔ {os.path.basename(pair[1])} | {str(e)}"
            if verbose:
                print(line)
            log_lines.append(line)

    if not overlaps:
        raise RuntimeError("No valid image pairs processed.")

    mode_overlap = Counter(overlaps).most_common(1)[0][0]
    median_mse = float(np.median(mses))
    elapsed = time.time() - start_time

    summary = [
        "\n GPU Optimized Processing Summary:",
        f"   Total pairs:   {total_pairs}",
        f"   Valid pairs:   {valid_pairs}",
        f"   Skipped pairs: {skipped_pairs}",
        f"   Elapsed time:  {elapsed:.2f}s",
        f"   Mode Overlap:  {mode_overlap}, Median MSE: {median_mse:.2f}"
    ]
    for line in summary:
        if verbose:
            print(line)
        log_lines.append(line)

    with open("stats.txt", "w", encoding="utf-8") as f:
        for line in log_lines:
            f.write(line + "\n")

    return mode_overlap, median_mse
    
def compute_all_overlaps_gpu(small_np, big_np):
    """
    Compute all 512 horizontal overlaps between two images using CuPy.
    Returns (best_overlap, best_mse).
    """
    small_cp = cp.asarray(small_np, dtype=cp.float32)
    big_cp = cp.asarray(big_np, dtype=cp.float32)
    h = small_cp.shape[1]

    # Store MSEs for each offset
    mses = cp.zeros(h, dtype=cp.float32)

    for offset in range(1, h + 1):
        small_slice = small_cp[:, -offset:]
        big_slice = big_cp[:, :offset]
        mse = cp.mean((small_slice - big_slice) ** 2)
        mses[offset - 1] = mse

    best_overlap = int(cp.argmin(mses).item())
    best_mse = float(mses[best_overlap].item())
    return best_overlap + 1, best_mse

import cupy as cp
import numpy as np
def compute_all_overlaps_gpu_vectorized(imggen):
    """
    Compute all horizontal overlaps (1 to 512) between many 512x512 grayscale image pairs using CuPy vectorization.
    For each pair, find the best overlap (lowest MSE), then return:
    - The mode of best overlaps
    - The average MSE for that mode only

    Parameters:
    - imggen: iterable of (small_img, big_img), both 512x512 grayscale numpy arrays

    Returns:
    - mode_overlap (int): most frequent best overlap value
    - avg_mse_for_mode (float): average MSE among pairs with that mode
    """
    from collections import defaultdict, Counter

    best_overlaps = []
    mse_by_overlap = defaultdict(list)

    for small_np, big_np in imggen:
        assert small_np.shape == (512, 512)
        assert big_np.shape == (512, 512)

        small = cp.asarray(small_np, dtype=cp.float32)
        big = cp.asarray(big_np, dtype=cp.float32)

        mses = []

        for offset in range(1, 513):
            s = small[:, -offset:]
            b = big[:, :offset]
            pad_width = 512 - offset

            s_padded = cp.pad(s, ((0, 0), (pad_width, 0)), mode='constant')
            b_padded = cp.pad(b, ((0, 0), (pad_width, 0)), mode='constant')

            mse = cp.mean((s_padded - b_padded) ** 2)
            mses.append(mse)

        mses = cp.stack(mses)
        best_idx = int(cp.argmin(mses).item())  # 0-based
        best_overlap = best_idx + 1
        best_mse = float(mses[best_idx].item())

        best_overlaps.append(best_overlap)
        mse_by_overlap[best_overlap].append(best_mse)

    if not best_overlaps:
        raise RuntimeError("No overlaps were computed.")

    mode_overlap = Counter(best_overlaps).most_common(1)[0][0]
    mse_list = mse_by_overlap[mode_overlap]
    avg_mse_for_mode = sum(mse_list) / len(mse_list)

    return mode_overlap, avg_mse_for_mode


def test_overlap():
    # Replace with your actual image paths
    path_small = r"E:\OCT\chrysopt2-2025-07-08\averaged_images\average\511-00_avg.tif"
    path_big = r"E:\OCT\chrysopt2-2025-07-08\averaged_images\average\511-01_avg.tif"

    small = load_grayscale_image(path_small)
    big = load_grayscale_image(path_big)

    if small.shape != big.shape:
        raise ValueError("Images must be the same shape (512x512 assumed)")

    best_overlap, best_mse = compute_overlap_mse(small, big)

    print(f" Median overlap: {best_overlap} pixels (MSE: {best_mse:.2f})")

def test_overlap_generator():
    # Replace with your actual image paths
    folder_path = r"E:\OCT\chrysopt2-2025-07-08\averaged_images\average"
    
    gen = image_pairs_by_v(folder_path)
    median_overlap, median_mse = compute_overlap_from_generator(gen)
    
    print(f" Median overlap: {median_overlap} pixels (MSE: {median_mse:.2f})")


def compute_black_overlap_stats(img1, img2):
    """
    Returns the percentage of:
    - black pixels in img1
    - black pixels in img2
    - black pixels that are black in both
    """
    mask1 = img1 == 0
    mask2 = img2 == 0
    both_black = mask1 & mask2

    total = img1.size
    percent_black1 = 100 * np.sum(mask1) / total
    percent_black2 = 100 * np.sum(mask2) / total
    percent_both_black = 100 * np.sum(both_black) / total

    return percent_black1, percent_black2, percent_both_black

def get_optimal_overlap(folder):
    @contextmanager
    def silent_print():
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        try:
            yield
        finally:
            sys.stdout.close()
            sys.stdout = original_stdout
    
    image_gen = list(image_pairs_by_v(folder))
    # Vectorized GPU (requiere imágenes ya cargadas, no rutas)
    loaded_pairs = []
    for path_small, path_big in image_gen:
        img_a = load_grayscale_image(path_small)
        img_b = load_grayscale_image(path_big)
        loaded_pairs.append((img_a, img_b))

    with silent_print():
        start_vec = time.time()
        mode_overlap_vec, avg_mse_vec = compute_all_overlaps_gpu_vectorized(loaded_pairs)
        end_vec = time.time()
        print(f"[STITCHING] Overlap {mode_overlap_vec} MSE {avg_mse_vec} Time {end_vec-start_vec}")
        return mode_overlap_vec
class TestImageOverlapPerformance(unittest.TestCase):
    
            
    def setUp(self):
        self.folder = r"E:\OCT\chrysopt2-2025-07-08\averaged_images\average"
        self.image_gen1 = list(image_pairs_by_v(self.folder))  # list so we can reuse it
        self.image_gen2 = list(self.image_gen1)  # clone for the parallel test

    def test_compare_parallel_vs_sequential(self):
        # Sequential
        start_seq = time.time()
        median_overlap_seq, median_mse_seq = compute_overlap_from_generator(self.image_gen1)
        end_seq = time.time()
        duration_seq = end_seq - start_seq

        # Parallel
        start_par = time.time()
        median_overlap_par, median_mse_par = compute_overlap_parallel(self.image_gen2)
        end_par = time.time()
        duration_par = end_par - start_par

        print(f"\n🔍 Sequential   - Overlap: {median_overlap_seq}, Time: {duration_seq:.2f}s")
        print(f"⚡ Parallel     - Overlap: {median_overlap_par}, Time: {duration_par:.2f}s")

        # Ensure results are consistent
        self.assertEqual(median_overlap_seq, median_overlap_par)
        self.assertAlmostEqual(median_mse_seq, median_mse_par, places=2)

        # Ensure parallel version is not slower
        self.assertLessEqual(duration_par, duration_seq * 1.2)  # Allow some fluctuation
        
    def test_compare_gpu_vs_parallel(self):
        # Parallel CPU
        start_par = time.time()
        mode_overlap_par, median_mse_par = compute_overlap_parallel(self.image_gen1, verbose=False)
        end_par = time.time()
        duration_par = end_par - start_par

        # GPU
        start_gpu = time.time()
        mode_overlap_gpu, median_mse_gpu = compute_overlap_gpu(self.image_gen2, verbose=False)
        end_gpu = time.time()
        duration_gpu = end_gpu - start_gpu

        print(f"\n Parallel CPU - Mode Overlap: {mode_overlap_par}, Time: {duration_par:.2f}s")
        print(f" GPU Compute  - Mode Overlap: {mode_overlap_gpu}, Time: {duration_gpu:.2f}s")

        # Ensure results are consistent
        self.assertEqual(mode_overlap_par, mode_overlap_gpu)
        self.assertAlmostEqual(median_mse_par, median_mse_gpu, places=2)

        # Ensure GPU is not slower (allow fluctuation)
        self.assertLessEqual(duration_gpu, duration_par * 1.2)
    
    def test_compare_gpu_original_vs_optimized(self):
        # Original GPU
        start_gpu = time.time()
        mode_overlap_gpu, median_mse_gpu = compute_overlap_gpu(self.image_gen1, verbose=False)
        end_gpu = time.time()
        duration_gpu = end_gpu - start_gpu

        # Optimized GPU
        start_gpu_opt = time.time()
        mode_overlap_gpu_opt, median_mse_gpu_opt = compute_overlap_gpu_optimized(self.image_gen2, verbose=False)
        end_gpu_opt = time.time()
        duration_gpu_opt = end_gpu_opt - start_gpu_opt

        print(f"\n Original GPU       - Mode Overlap: {mode_overlap_gpu}, Time: {duration_gpu:.2f}s")
        print(f" Optimized GPU      - Mode Overlap: {mode_overlap_gpu_opt}, Time: {duration_gpu_opt:.2f}s")

        # Ensure same results
        self.assertEqual(mode_overlap_gpu, mode_overlap_gpu_opt)
        self.assertAlmostEqual(median_mse_gpu, median_mse_gpu_opt, places=2)

        # Ensure optimized version is not slower
        self.assertLessEqual(duration_gpu_opt, duration_gpu * 1.2)
        
    def test_compare_gpu_optimized_vs_vectorized(self):
        from contextlib import contextmanager
        import sys, os

        @contextmanager
        def silent_print():
            original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
            try:
                yield
            finally:
                sys.stdout.close()
                sys.stdout = original_stdout

        # Optimized GPU (con loop y rutas)
        with silent_print():
            start_opt = time.time()
            mode_overlap_opt, median_mse_opt = compute_overlap_gpu_optimized(self.image_gen1, verbose=False)
            end_opt = time.time()
        duration_opt = end_opt - start_opt

        # Vectorized GPU (requiere imágenes ya cargadas, no rutas)
        loaded_pairs = []
        for path_small, path_big in self.image_gen2:
            img_a = load_grayscale_image(path_small)
            img_b = load_grayscale_image(path_big)
            loaded_pairs.append((img_a, img_b))

        with silent_print():
            start_vec = time.time()
            mode_overlap_vec, avg_mse_vec = compute_all_overlaps_gpu_vectorized(loaded_pairs)
            end_vec = time.time()
        duration_vec = end_vec - start_vec

        print(f"\n Optimized GPU      - Mode Overlap: {mode_overlap_opt}, Time: {duration_opt:.2f}s")
        print(f" Vectorized GPU     - Mode Overlap: {mode_overlap_vec}, Time: {duration_vec:.2f}s")

        # Validación
        self.assertEqual(mode_overlap_opt, mode_overlap_vec)
        self.assertAlmostEqual(median_mse_opt, avg_mse_vec, places=2)

        self.assertLessEqual(duration_vec, duration_opt * 1.2)




import re
from collections import Counter
import matplotlib.pyplot as plt

def parse_and_plot_histogram(filepath, max_black_pct=95.0):
    """
    Parse the file and plot histogram of Overlaps where both black percentages are < max_black_pct.
    
    Parameters:
    - filepath: Path to the text file.
    - max_black_pct: Maximum allowed black percentage for both images.
    """
    pattern = re.compile(
        r"Overlap: (\d+), MSE: [\d.]+ \| Black %: small=([\d.]+)%, big=([\d.]+)%"
    )

    overlaps = []

    with open(filepath, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                overlap = int(match.group(1))
                black_small = float(match.group(2))
                black_big = float(match.group(3))

                if black_small < max_black_pct and black_big < max_black_pct:
                    overlaps.append(overlap)

    # Count frequency of each overlap value
    counter = Counter(overlaps)

    # Plot histogram
    plt.figure(figsize=(10, 5))
    plt.bar(counter.keys(), counter.values(), width=1.0)
    plt.title(f"Histogram of Overlaps (Black % < {max_black_pct}%)")
    plt.xlabel("Overlap")
    plt.ylabel("Frequency")
    plt.xlim(0, 512)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    return counter

# Example usage:
import sys
import os
from contextlib import contextmanager

@contextmanager
def silent_print():
    """
    Context manager to suppress all stdout prints (e.g., during benchmarking).
    """
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        yield
    finally:
        sys.stdout.close()
        sys.stdout = original_stdout



       
#if __name__ == '__main__':
    #unittest.main()
#    parse_and_plot_histogram("stats.txt", max_black_pct=92.0)
if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(TestImageOverlapPerformance('test_compare_gpu_optimized_vs_vectorized'))
    unittest.TextTestRunner().run(suite)


