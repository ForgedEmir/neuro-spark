import requests
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

BASE = "https://physionet.org/files/eegmmidb/1.0.0/"
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "eeg") + "/"
WORKERS = int(os.environ.get("DOWNLOAD_WORKERS", 10))
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds


def fetch_records():
    """Fetch the RECORDS file from PhysioNet with retry."""
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(BASE + "RECORDS", timeout=30)
            r.raise_for_status()
            return [l.strip() for l in r.text.splitlines() if l.strip()]
        except requests.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                raise SystemExit(f"Failed to fetch RECORDS after {MAX_RETRIES} attempts: {e}")


def download_one(rec):
    """Download a single record with retry and resume support."""
    dest = os.path.join(OUT, rec)
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    # Check if already fully downloaded
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return "skip", rec

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(BASE + rec, timeout=120, stream=True)
            resp.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)
            return "ok", rec
        except requests.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
                # Clean partial file before retry
                if os.path.exists(dest):
                    os.remove(dest)
            else:
                # Clean failed file
                if os.path.exists(dest):
                    os.remove(dest)
                return "error", f"{rec}: {e}"


def main():
    records = fetch_records()

    # Filter already downloaded
    todo = [rec for rec in records
            if not (os.path.exists(os.path.join(OUT, rec))
                    and os.path.getsize(os.path.join(OUT, rec)) > 0)]

    already_done = len(records) - len(todo)
    print(f"Total: {len(records)} | Already done: {already_done} | To download: {len(todo)}", flush=True)

    if not todo:
        print("All records already downloaded.")
        return

    downloaded = skipped = errors = 0

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(download_one, rec): rec for rec in todo}
        with tqdm(total=len(todo), desc="Downloading EEG files", unit="file") as pbar:
            for future in as_completed(futures):
                status, info = future.result()
                if status == "ok":
                    downloaded += 1
                elif status == "skip":
                    skipped += 1
                else:
                    errors += 1
                    tqdm.write(f"ERROR: {info}")
                pbar.update(1)

    print(f"\nDone — downloaded={downloaded}, skipped={skipped}, errors={errors}")


if __name__ == "__main__":
    main()
