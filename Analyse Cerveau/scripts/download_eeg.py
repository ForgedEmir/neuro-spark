import requests, os
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = "https://physionet.org/files/eegmmidb/1.0.0/"
OUT  = os.path.join(os.path.dirname(__file__), "..", "data", "eeg") + "/"
WORKERS = 10  # parallel downloads

r = requests.get(BASE + "RECORDS", timeout=30)
records = [l.strip() for l in r.text.splitlines() if l.strip()]

# Filter only records not yet downloaded
todo = []
for rec in records:
    dest = os.path.join(OUT, rec)
    if not (os.path.exists(dest) and os.path.getsize(dest) > 0):
        todo.append(rec)

print(f"Total: {len(records)} | Already done: {len(records)-len(todo)} | Remaining: {len(todo)}", flush=True)

downloaded = skipped = errors = 0
total_done = len(records) - len(todo)

def download_one(rec):
    dest = os.path.join(OUT, rec)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        resp = requests.get(BASE + rec, timeout=60, stream=True)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)
        return "ok", rec
    except Exception as e:
        return "error", f"{rec}: {e}"

with ThreadPoolExecutor(max_workers=WORKERS) as executor:
    futures = {executor.submit(download_one, rec): rec for rec in todo}
    for future in as_completed(futures):
        status, info = future.result()
        total_done += 1
        if status == "ok":
            downloaded += 1
        else:
            errors += 1
            print(f"ERROR {info}", flush=True)
        if total_done % 50 == 0:
            print(f"[{total_done}/{len(records)}] downloaded={downloaded} errors={errors}", flush=True)

print(f"\nDone — downloaded={downloaded}, errors={errors}")
