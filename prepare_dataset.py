"""
Dataset Preparation Script for Indian Currency Note Classification
===================================================================
Prepares a lightweight, clean, and balanced subset (50 images per class)
from the Mendeley Indian Currency Dataset for building a CNN from scratch
using pure Python and NumPy.

Author: Antigravity
"""

import os
import sys
import json
import time
import hashlib
import argparse
import urllib.request
import urllib.error
import ssl
from datetime import datetime
from pathlib import Path
from PIL import Image

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
DEFAULT_DATASET_URL = "https://data.mendeley.com/datasets/48ympv8jjf/1"
DEFAULT_DATASET_ID = "48ympv8jjf"
DEFAULT_DATASET_VERSION = "1"
DEFAULT_OUTPUT_DIR = "dataset"
DEFAULT_METADATA_FILE = "dataset_metadata.json"
IMAGES_PER_CLASS = 50
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


# -----------------------------------------------------------------------------
# Helper Utilities
# -----------------------------------------------------------------------------
def sanitize_class_name(name: str) -> str:
    """Standardizes class names (e.g. 'Rs.10' -> 'Rs_10') for file system compatibility."""
    return name.strip().replace(".", "_").replace(" ", "_")


def compute_sha256(data: bytes) -> str:
    """Computes SHA-256 hash of byte content."""
    return hashlib.sha256(data).hexdigest()


def print_progress_bar(iteration: int, total: int, prefix: str = '', suffix: str = '', length: int = 30):
    """Draws a clean terminal progress bar."""
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = '=' * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f"\r{prefix} |{bar}| {iteration}/{total} ({percent}%) {suffix}")
    sys.stdout.flush()
    if iteration == total:
        sys.stdout.write("\n")


# -----------------------------------------------------------------------------
# Dataset Preparation Class
# -----------------------------------------------------------------------------
class CurrencyDatasetPreparator:
    def __init__(
        self,
        dataset_url: str = DEFAULT_DATASET_URL,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        images_per_class: int = IMAGES_PER_CLASS,
        metadata_file: str = DEFAULT_METADATA_FILE
    ):
        self.dataset_url = dataset_url
        self.dataset_id = self._extract_dataset_id(dataset_url)
        self.version = self._extract_version(dataset_url)
        self.output_dir = Path(output_dir)
        self.images_per_class = images_per_class
        self.metadata_file = Path(metadata_file)
        
        # SSL Context for HTTPS requests
        self.ssl_ctx = ssl.create_default_context()
        
        # Tracking metrics for the final summary
        self.stats = {
            "classes_found": 0,
            "classes_list": [],
            "target_per_class": images_per_class,
            "total_requested": 0,
            "total_downloaded": 0,
            "total_cached": 0,
            "total_processed_success": 0,
            "total_corrupted_or_failed": 0,
            "total_duplicates": 0,
            "class_breakdown": {},
            "errors": []
        }
        
        # Global set to track duplicate image hashes across the dataset
        self.seen_hashes = set()

    def _extract_dataset_id(self, url: str) -> str:
        parts = url.rstrip("/").split("/")
        for i, part in enumerate(parts):
            if part == "datasets" and i + 1 < len(parts):
                return parts[i + 1]
        return DEFAULT_DATASET_ID

    def _extract_version(self, url: str) -> str:
        parts = url.rstrip("/").split("/")
        if len(parts) >= 2 and parts[-2] == self.dataset_id:
            return parts[-1]
        return DEFAULT_DATASET_VERSION

    def _make_api_request(self, url: str, custom_accept: str = None) -> any:
        """Executes a GET request against Mendeley Data endpoints with proper headers."""
        headers = {
            "User-Agent": USER_AGENT,
            "Accept": custom_accept or "application/json, text/plain, */*",
            "Referer": self.dataset_url
        }
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, context=self.ssl_ctx, timeout=20) as resp:
            data = resp.read()
            return json.loads(data.decode("utf-8"))

    def fetch_dataset_structure(self) -> list:
        """
        Discovers all class categories/folders in the dataset.
        Returns a list of folder dictionaries: [{'id': ..., 'name': ...}, ...]
        """
        print(f"\n[Step 1/4] Discovering dataset structure from: {self.dataset_url}")
        folders_url = f"https://data.mendeley.com/public-api/datasets/{self.dataset_id}/folders/{self.version}"
        
        try:
            folders = self._make_api_request(
                folders_url,
                custom_accept="application/vnd.mendeley-public-dataset.1+json"
            )
        except Exception as e:
            print(f"[!] Warning: Public folders endpoint error ({e}). Attempting fallback...")
            # Fallback direct inspection
            folders = [
                {"id": "79106392-976a-470e-aaf0-b84621794b54", "name": "Rs.10"},
                {"id": "bd4d0b7b-b3c5-41a3-999d-54812a6a0081", "name": "Rs.20"},
                {"id": "4e900bc4-d7f3-45d4-a159-c9c4bc55850d", "name": "Rs.50"},
                {"id": "90b73257-aaa9-4c39-a803-189f262d04c4", "name": "Rs.100"},
                {"id": "17b304df-a5bd-4c8a-9fdb-9bc72de3845e", "name": "Rs.200"},
                {"id": "ccb77215-1c70-4228-aaf8-60c1717a83e5", "name": "Rs.500"},
                {"id": "6934acc5-a44a-48fa-8fa5-5337d68f36e8", "name": "Rs.2000"}
            ]

        # Sort folders by denomination amount for consistent, intuitive ordering
        def parse_denomination(f):
            name = f.get("name", "")
            digits = "".join(ch for ch in name if ch.isdigit())
            return int(digits) if digits else 999999

        folders = sorted(folders, key=parse_denomination)
        
        self.stats["classes_found"] = len(folders)
        self.stats["classes_list"] = [f.get("name") for f in folders]
        self.stats["total_requested"] = len(folders) * self.images_per_class
        
        print(f"[*] Found {len(folders)} currency classes:")
        for idx, f in enumerate(folders, 1):
            print(f"    {idx}. {f.get('name')} (Folder ID: {f.get('id')})")
            
        return folders

    def fetch_folder_file_catalog(self, folder_id: str, folder_name: str) -> list:
        """
        Retrieves the catalog of files available for a given folder/class.
        Sorts the filenames deterministically.
        """
        url = (
            f"https://data.mendeley.com/public-api/datasets/{self.dataset_id}/files"
            f"?folder_id={folder_id}&version={self.version}&$start=0&$limit=1000"
        )
        try:
            files = self._make_api_request(
                url,
                custom_accept="application/vnd.mendeley-public-dataset.1+json"
            )
            if isinstance(files, dict) and "error" in files:
                raise ValueError(f"API returned error: {files}")
        except Exception:
            # Fallback for folders that encounter server-side public-api issues (e.g. Rs.100)
            url_fallback = f"https://data.mendeley.com/api/datasets/{self.dataset_id}/files"
            all_files = self._make_api_request(url_fallback)
            files = [f for f in all_files if f.get("folder_id") == folder_id]

        # Deterministic sorting by filename (alphanumeric) to guarantee non-random, reproducible selection
        files = sorted(files, key=lambda x: x.get("filename", ""))
        return files

    def _download_single_image(self, download_url: str, max_retries: int = 3) -> bytes:
        """Downloads image bytes with retries and timeout protection."""
        headers = {
            "User-Agent": USER_AGENT,
            "Referer": self.dataset_url
        }
        for attempt in range(1, max_retries + 1):
            try:
                req = urllib.request.Request(download_url, headers=headers)
                with urllib.request.urlopen(req, context=self.ssl_ctx, timeout=25) as resp:
                    if resp.status == 200:
                        return resp.read()
            except Exception as e:
                if attempt == max_retries:
                    raise RuntimeError(f"Failed after {max_retries} attempts: {e}")
                time.sleep(1.0 * attempt)
        raise RuntimeError("Download failed.")

    def _validate_image(self, img_bytes: bytes) -> tuple:
        """
        Validates image data using PIL.
        Checks for corruption, truncated data, and extracts dimensions.
        Returns: (is_valid, width, height, format_name, error_msg)
        """
        try:
            import io
            # 1. Verification of structural integrity
            bio = io.BytesIO(img_bytes)
            img = Image.open(bio)
            img.verify()
            
            # 2. Re-open to inspect actual pixel data & dimensions (verify() invalidates image object)
            bio.seek(0)
            img = Image.open(bio)
            img.load()  # Forces full decode to catch truncation
            width, height = img.size
            format_name = img.format or "JPEG"
            
            # Ensure 3-channel RGB convertible
            if img.mode not in ("RGB", "L", "RGBA"):
                return False, 0, 0, "", f"Unsupported image mode: {img.mode}"
                
            return True, width, height, format_name, ""
        except Exception as e:
            return False, 0, 0, "", str(e)

    def prepare(self) -> dict:
        """
        Main execution workflow:
        1. Discover classes
        2. Create directories
        3. Download & process first 50 valid images per class
        4. Save metadata
        5. Print summary
        """
        start_time = time.time()
        
        # 1. Discover classes
        folders = self.fetch_dataset_structure()
        
        # 2. Setup base output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[Step 2/4] Target dataset directory: {self.output_dir.resolve()}")
        
        metadata_records = {
            "dataset_info": {
                "name": "Indian Currency Dataset (Curated 50-per-class)",
                "source_url": self.dataset_url,
                "dataset_id": self.dataset_id,
                "created_at": datetime.now().isoformat(),
                "description": "50 images per currency denomination for CNN from scratch training.",
                "selection_policy": "First 50 valid images per class ordered lexicographically by original filename."
            },
            "class_to_idx": {},
            "idx_to_class": {},
            "classes": [],
            "images": []
        }

        # Build class index mapping
        for idx, f in enumerate(folders):
            cname = sanitize_class_name(f.get("name"))
            metadata_records["class_to_idx"][cname] = idx
            metadata_records["idx_to_class"][str(idx)] = cname
            metadata_records["classes"].append(cname)

        print(f"\n[Step 3/4] Downloading and validating exactly {self.images_per_class} images per class...")
        
        total_selected = 0

        for class_idx, folder in enumerate(folders, 1):
            raw_cname = folder.get("name")
            cname = sanitize_class_name(raw_cname)
            folder_id = folder.get("id")
            
            class_dir = self.output_dir / cname
            class_dir.mkdir(parents=True, exist_ok=True)
            
            # Fetch catalog for this class
            file_candidates = self.fetch_folder_file_catalog(folder_id, raw_cname)
            
            self.stats["class_breakdown"][cname] = {
                "available_in_source": len(file_candidates),
                "selected": 0,
                "downloaded": 0,
                "cached": 0,
                "corrupted": 0,
                "duplicates": 0
            }
            
            valid_images_for_class = 0
            candidate_idx = 0
            
            print(f"\n-> Processing Class [{class_idx}/{len(folders)}]: '{cname}' (Available: {len(file_candidates)})")
            
            while valid_images_for_class < self.images_per_class and candidate_idx < len(file_candidates):
                file_obj = file_candidates[candidate_idx]
                candidate_idx += 1
                
                orig_filename = file_obj.get("filename", f"file_{candidate_idx}.jpg")
                content_details = file_obj.get("content_details", {})
                download_url = content_details.get("download_url")
                
                target_filename = f"image_{valid_images_for_class + 1:03d}.jpg"
                target_filepath = class_dir / target_filename
                
                # Check if file already exists locally and is valid (reproducibility / caching)
                img_bytes = None
                is_cached = False
                
                if target_filepath.exists():
                    try:
                        with open(target_filepath, "rb") as f:
                            img_bytes = f.read()
                        is_valid, w, h, fmt, err = self._validate_image(img_bytes)
                        if is_valid:
                            is_cached = True
                            self.stats["total_cached"] += 1
                            self.stats["class_breakdown"][cname]["cached"] += 1
                    except Exception:
                        img_bytes = None

                # Download if not cached
                if not is_cached:
                    if not download_url:
                        self.stats["total_corrupted_or_failed"] += 1
                        self.stats["class_breakdown"][cname]["corrupted"] += 1
                        self.stats["errors"].append(f"Missing download URL for {orig_filename}")
                        continue
                        
                    try:
                        img_bytes = self._download_single_image(download_url)
                        self.stats["total_downloaded"] += 1
                        self.stats["class_breakdown"][cname]["downloaded"] += 1
                    except Exception as e:
                        self.stats["total_corrupted_or_failed"] += 1
                        self.stats["class_breakdown"][cname]["corrupted"] += 1
                        self.stats["errors"].append(f"Download error on {orig_filename}: {e}")
                        continue

                # Validate image integrity
                is_valid, width, height, format_name, error_msg = self._validate_image(img_bytes)
                if not is_valid:
                    self.stats["total_corrupted_or_failed"] += 1
                    self.stats["class_breakdown"][cname]["corrupted"] += 1
                    self.stats["errors"].append(f"Corrupted image {orig_filename}: {error_msg}")
                    # Delete corrupted local file if exists
                    if target_filepath.exists():
                        target_filepath.unlink(missing_ok=True)
                    continue

                # Duplicate detection via SHA-256
                img_sha256 = compute_sha256(img_bytes)
                if img_sha256 in self.seen_hashes and not is_cached:
                    self.stats["total_duplicates"] += 1
                    self.stats["class_breakdown"][cname]["duplicates"] += 1
                    self.stats["errors"].append(f"Duplicate image skipped: {orig_filename} (SHA: {img_sha256[:8]})")
                    continue
                
                self.seen_hashes.add(img_sha256)

                # Write to disk if downloaded fresh
                if not is_cached:
                    with open(target_filepath, "wb") as f:
                        f.write(img_bytes)

                valid_images_for_class += 1
                self.stats["total_processed_success"] += 1
                self.stats["class_breakdown"][cname]["selected"] += 1
                total_selected += 1

                # Record metadata
                metadata_records["images"].append({
                    "id": total_selected,
                    "class_name": cname,
                    "class_label": metadata_records["class_to_idx"][cname],
                    "relative_path": f"{cname}/{target_filename}",
                    "original_filename": orig_filename,
                    "sha256": img_sha256,
                    "width": width,
                    "height": height,
                    "format": format_name
                })

                # Display progress
                status_tag = "Cached" if is_cached else "Downloaded"
                print_progress_bar(
                    valid_images_for_class,
                    self.images_per_class,
                    prefix=f"  [{cname}]",
                    suffix=f"{target_filename} ({status_tag})"
                )

        # 4. Save metadata file
        print(f"\n[Step 4/4] Writing metadata file to: {self.metadata_file.resolve()}")
        metadata_records["summary"] = {
            "total_images": len(metadata_records["images"]),
            "classes_count": len(metadata_records["classes"]),
            "images_per_class": self.images_per_class,
            "classes": metadata_records["classes"]
        }
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata_records, f, indent=2)

        elapsed = time.time() - start_time
        
        # 5. Output rich summary
        self._print_summary(elapsed)
        return self.stats

    def _print_summary(self, elapsed_seconds: float):
        """Prints the final formatted summary table."""
        print("\n" + "=" * 65)
        print("                 DATASET PREPARATION SUMMARY")
        print("=" * 65)
        print(f" Dataset URL                     : {self.dataset_url}")
        print(f" Local Dataset Directory         : {self.output_dir.resolve()}")
        print(f" Metadata File                   : {self.metadata_file.resolve()}")
        print(f" Total Number of Classes Found   : {self.stats['classes_found']}")
        print(f" Target Images per Class         : {self.stats['target_per_class']}")
        print(f" Total Target Images Requested   : {self.stats['total_requested']}")
        print(f" Successfully Processed Images   : {self.stats['total_processed_success']}")
        print(f" Freshly Downloaded Images       : {self.stats['total_downloaded']}")
        print(f" Verified Cached Images          : {self.stats['total_cached']}")
        print(f" Failed / Corrupted Images       : {self.stats['total_corrupted_or_failed']}")
        print(f" Duplicate Images Skipped        : {self.stats['total_duplicates']}")
        print(f" Time Taken                      : {elapsed_seconds:.2f} seconds")
        print("-" * 65)
        print(f"{'Class Name':<15} | {'Available':<10} | {'Selected':<10} | {'Status':<10}")
        print("-" * 65)
        for cname, info in self.stats["class_breakdown"].items():
            status_str = "COMPLETE" if info["selected"] == self.images_per_class else "PARTIAL"
            print(f"{cname:<15} | {info['available_in_source']:<10} | {info['selected']:<10} | {status_str:<10}")
        print("=" * 65)
        
        if self.stats["errors"]:
            print(f"\n[!] Notice: {len(self.stats['errors'])} anomalies detected and handled gracefully.")
            for err in self.stats["errors"][:5]:
                print(f"    - {err}")
            if len(self.stats["errors"]) > 5:
                print(f"    ... and {len(self.stats['errors']) - 5} more.")
        else:
            print("\n[+] All images verified with 100% integrity without corruption or missing files.")
        print("\n[SUCCESS] Data preparation complete. Ready for NumPy CNN training pipeline.\n")


# -----------------------------------------------------------------------------
# Main CLI Entry Point
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare 50 images per currency class for NumPy CNN training."
    )
    parser.add_argument(
        "--url",
        type=str,
        default=DEFAULT_DATASET_URL,
        help=f"Mendeley Dataset URL (default: {DEFAULT_DATASET_URL})"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Target directory to store images (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--images-per-class",
        type=int,
        default=IMAGES_PER_CLASS,
        help=f"Number of images to select per class (default: {IMAGES_PER_CLASS})"
    )
    parser.add_argument(
        "--metadata-file",
        type=str,
        default=DEFAULT_METADATA_FILE,
        help=f"Output metadata JSON path (default: {DEFAULT_METADATA_FILE})"
    )
    
    args = parser.parse_args()
    
    preparator = CurrencyDatasetPreparator(
        dataset_url=args.url,
        output_dir=args.output_dir,
        images_per_class=args.images_per_class,
        metadata_file=args.metadata_file
    )
    preparator.prepare()


if __name__ == "__main__":
    main()
