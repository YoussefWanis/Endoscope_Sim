"""
install_dataset.py  —  C3VD / C3VDv2 Dataset Installer
========================================================
Downloads one or more C3VD video sequences from Google Drive and unpacks
them into the correct folder structure for the endoscope app.

Usage:
    python install_dataset.py               # interactive menu
    python install_dataset.py --small       # download only the 2 smallest sequences (~1.8 GB total)
    python install_dataset.py --list        # list all available sequences
    python install_dataset.py --check       # check what is already installed

The app looks for data here:
    data/c3vd/<sequence_name>/NNNN.png            ← video frames
    data/c3vd/<sequence_name>/depth/NNNN_depth.tiff
    data/c3vd/<sequence_name>/pose.txt

If files are not found the app will still run in demo mode with synthetic frames.
"""

import os, sys, zipfile, shutil, json, ssl, urllib.request, urllib.error, argparse, time
import http.cookiejar, re

# ── paths ────────────────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE, 'data', 'c3vd')
CACHE_DIR = os.path.join(BASE, 'data', '_cache')

# ── dataset catalogue (C3VD v1, CC BY-NC-SA 4.0) ────────────────────────────
# Google Drive direct-download links from durrlab.github.io/C3VD
SEQUENCES = [
    # (name, gdrive_id, size_GB, region, texture, video)
    ('trans_t1_a',   '1urFuVo8ZalwPmsXEZg3xzhuqhpgWV8hw', 0.59,  'Transverse Colon', 1, 'a'),
    ('trans_t2_a',   '1ylZWWtVlXfDx9dhPIeWJ1HqDHJZ3QKdH', 1.58,  'Transverse Colon', 2, 'a'),
    ('trans_t2_b',   '1vru228_TEgxT3aS90CmvOWsMB0RLnAxn', 0.97,  'Transverse Colon', 2, 'b'),
    ('trans_t2_c',   '12YpowbP6zhoO_Qx9UBwhfRLXNJN1EAu4', 1.83,  'Transverse Colon', 2, 'c'),
    ('trans_t3_a',   '1B4aeZfAqmUJgWr8e-2YAibUe4er30ncr', 1.83,  'Transverse Colon', 3, 'a'),
    ('trans_t3_b',   '1ZpbYcDVP-sCTjjQrDc303olFgsr2nA5J', 1.66,  'Transverse Colon', 3, 'b'),
    ('trans_t4_a',   '18qzXMifS54jAx29yROKXXxxZg0qo-iTz', 3.10,  'Transverse Colon', 4, 'a'),
    ('trans_t4_b',   '1C-nw6MR7sxssw3LS-GpiPmwBzEYhUCHN', 4.61,  'Transverse Colon', 4, 'b'),
    ('desc_t4_a',    '1d9HDNg4-Og1cTWM-eIU5SWM2BMrMWhwQ', 1.24,  'Descending Colon', 4, 'a'),
    ('sigmoid_t1_a', '19VGDuZ73OWNwM8eIgDkkZYBPJPQ5BD91', 5.20,  'Sigmoid Colon',    1, 'a'),
    ('sigmoid_t2_a', '16epAys428g9vBQgm611TElMyAXORo7rH', 4.22,  'Sigmoid Colon',    2, 'a'),
    ('sigmoid_t3_a', '1ZRU2KuHoc2XCbKSY_A1S-7BxEfKf9xPr', 4.58,  'Sigmoid Colon',    3, 'a'),
    ('sigmoid_t3_b', '1XfZFAQ5_Wxle8d5wSlOCumKSg4IP8wTv', 4.21,  'Sigmoid Colon',    3, 'b'),
    ('cecum_t1_a',   '14o6_4GQLZWx5dQq2L_drzmN_rlCT7Yhr', 2.86,  'Cecum',            1, 'a'),
    ('cecum_t1_b',   '1z3AHdnBH_YoCMnnTfDa8SNPQYIsvaBO3', 8.36,  'Cecum',            1, 'b'),
    ('cecum_t2_a',   '13XhJIev9memFtwUf_dnjJ7o8z6O_c-xW', 3.71,  'Cecum',            2, 'a'),
    ('cecum_t2_b',   '1ykYtQGiFesev5QLfz_avYuQ5a7Zs8kgF', 11.06, 'Cecum',            2, 'b'),
    ('cecum_t2_c',   '1tNoBLpPbrQexKlnOKMK2peERn9Rj_9Dp', 6.13,  'Cecum',            2, 'c'),
    ('cecum_t3_a',   '1Uw8uCRRDm_RrgkccGbiBXZHf9P-THM2Q', 6.80,  'Cecum',            3, 'a'),
    ('cecum_t4_a',   '1FC-dR__0LVb7WH02KpUx9TZVvvvN-Gyx', 5.04,  'Cecum',            4, 'a'),
    ('cecum_t4_b',   '11SbH2AZsuciTu3iGxdXCdQZky6uDTyS5', 4.41,  'Cecum',            4, 'b'),
    ('trans_t1_b',   '1hyjmd7vn86McE1nUnYCzvOm8LlyHLYwt', 5.07,  'Transverse Colon', 1, 'b'),
]

SMALL_SEQS = ['trans_t1_a', 'trans_t2_b']   # ~1.56 GB combined


# ── helpers ──────────────────────────────────────────────────────────────────

def bar(done, total, width=40):
    pct  = done / total if total else 0
    fill = int(pct * width)
    return f"[{'█'*fill}{'░'*(width-fill)}] {pct*100:5.1f}%  {done/1e6:.1f}/{total/1e6:.1f} MB"


def gdrive_url(file_id):
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def download_gdrive(file_id, dest_path, name):
    ssl_ctx = ssl._create_unverified_context()
    url     = gdrive_url(file_id)

    print(f"\n  Connecting to Google Drive for {name}…")
    try:
        cj = http.cookiejar.CookieJar()
        opener = urllib.request.build_opener(
            urllib.request.HTTPRedirectHandler(),
            urllib.request.HTTPCookieProcessor(cj),
            urllib.request.HTTPSHandler(context=ssl_ctx)
        )
        
        req  = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        resp = opener.open(req, timeout=30)

        token = None
        for cookie in cj:
            if cookie.name.startswith('download_warning'):
                token = cookie.value
                break
        
        if not token and resp.info().get_content_type() == 'text/html':
            html = resp.read().decode('utf-8', errors='ignore')
            m = re.search(r'confirm=([a-zA-Z0-9_-]+)', html)
            if m:
                token = m.group(1)

        if token:
            url2  = url + f"&confirm={token}"
            req2  = urllib.request.Request(url2, headers={'User-Agent': 'Mozilla/5.0'})
            resp  = opener.open(req2, timeout=30)

        if resp.info().get_content_type() == 'text/html':
            print("\n  ✗ Download failed: Google Drive requires manual confirmation or quota exceeded.")
            print("    Please use the manual installation method (Option 5).")
            return False

        total_size = int(resp.headers.get('Content-Length', 0))
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        done = 0
        t0   = time.time()
        with open(dest_path, 'wb') as f:
            while True:
                chunk = resp.read(1 << 17)   # 128 KB
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                elapsed = time.time() - t0
                speed   = done / elapsed / 1e6 if elapsed > 0 else 0
                sys.stdout.write(f"\r  {bar(done, total_size)}  {speed:.1f} MB/s")
                sys.stdout.flush()
        print()
        return True

    except urllib.error.URLError as e:
        print(f"\n  ✗ Download failed: {e}")
        return False
    except KeyboardInterrupt:
        print("\n  Download cancelled.")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False


def extract(zip_path, out_dir, seq_name):
    print(f"  Extracting {os.path.basename(zip_path)}…")
    seq_dir = os.path.join(out_dir, seq_name)
    os.makedirs(seq_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        members = zf.namelist()
        total   = len(members)
        for i, m in enumerate(members, 1):
            zf.extract(m, seq_dir)
            sys.stdout.write(f"\r  Extracting {i}/{total} files…")
            sys.stdout.flush()
    print(f"\n  Extracted to: {seq_dir}")
    # flatten one extra level if zip extracted into a subdir
    sub = os.path.join(seq_dir, seq_name)
    if os.path.isdir(sub):
        for item in os.listdir(sub):
            shutil.move(os.path.join(sub, item), os.path.join(seq_dir, item))
        os.rmdir(sub)


def is_installed(seq_name):
    rgb_dir = os.path.join(DATA_DIR, seq_name, 'rgb')
    if not os.path.isdir(rgb_dir):
        rgb_dir = os.path.join(DATA_DIR, seq_name)
    return os.path.isdir(rgb_dir) and len([f for f in os.listdir(rgb_dir) if f.endswith('.png')]) > 0


def installed_sequences():
    if not os.path.isdir(DATA_DIR):
        return []
    return [s for s in os.listdir(DATA_DIR) if is_installed(s)]


def frame_count(seq_name):
    rgb_dir = os.path.join(DATA_DIR, seq_name, 'rgb')
    if not os.path.isdir(rgb_dir):
        rgb_dir = os.path.join(DATA_DIR, seq_name)
    if not os.path.isdir(rgb_dir):
        return 0
    return len([f for f in os.listdir(rgb_dir) if f.endswith('.png')])


def install_sequence(seq):
    name, gid, size_gb, region, tex, vid = seq
    print(f"\n{'─'*60}")
    print(f"  Sequence : {name}")
    print(f"  Region   : {region}  Texture {tex}  Video {vid}")
    print(f"  Size     : ~{size_gb:.2f} GB")

    if is_installed(name):
        print(f"  Status   : ✓ already installed ({frame_count(name)} frames)")
        return True

    os.makedirs(CACHE_DIR, exist_ok=True)
    zip_path = os.path.join(CACHE_DIR, f"{name}.zip")

    ok = download_gdrive(gid, zip_path, name)
    if not ok:
        return False

    extract(zip_path, DATA_DIR, name)
    os.remove(zip_path)

    fc = frame_count(name)
    print(f"  ✓ Done  ({fc} frames installed)")
    return True


# ── manual-install instructions ───────────────────────────────────────────────

MANUAL_INSTRUCTIONS = """
╔══════════════════════════════════════════════════════════════════════════════╗
║            C3VD DATASET — MANUAL INSTALLATION GUIDE                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  The dataset is available from:                                              ║
║    https://durrlab.github.io/C3VD/                                           ║
║                                                                              ║
║  STEP 1 — Download one or more sequence zips from the page above.           ║
║            The smallest ones are:                                            ║
║              trans_t1_a.zip   (~0.59 GB)   61 frames                        ║
║              trans_t2_b.zip   (~0.97 GB)  103 frames                        ║
║                                                                              ║
║  STEP 2 — Extract each zip. You will get a folder like:                     ║
║              trans_t1_a/                                                     ║
║                0001.png, 0002.png, …                                         ║
║                depth/       ← 0001_depth.tiff, …                            ║
║                normals/                                                      ║
║                pose.txt                                                      ║
║                                                                              ║
║  STEP 3 — Place the extracted folder inside:                                 ║
║              <this project folder>/data/c3vd/                                ║
║                                                                              ║
║            Final structure should look like:                                 ║
║              c3vd_scope/                                                     ║
║              └── data/                                                       ║
║                  └── c3vd/                                                   ║
║                      └── trans_t1_a/                                         ║
║                          ├── 0001.png                                        ║
║                          ├── …                                               ║
║                          ├── depth/                                          ║
║                          └── pose.txt                                        ║
║                                                                              ║
║  STEP 4 — Run the app:                                                       ║
║              python endoscope_app.py                                         ║
║                                                                              ║
║  The app will auto-detect installed sequences on startup.                    ║
║  If no data is found it runs in DEMO mode with synthetic frames.             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


# ── CLI ───────────────────────────────────────────────────────────────────────

def cmd_list():
    print("\n  Available C3VD sequences:\n")
    print(f"  {'#':<3} {'Name':<20} {'Region':<20} {'Tex':>3} {'Vid':>3} {'Size':>7}  {'Status'}")
    print(f"  {'─'*70}")
    for i, seq in enumerate(SEQUENCES, 1):
        name, _, size, region, tex, vid = seq
        status = '✓ installed' if is_installed(name) else '─'
        print(f"  {i:<3} {name:<20} {region:<20} {tex:>3} {vid:>3} {size:>6.2f}GB  {status}")
    print()


def cmd_check():
    inst = installed_sequences()
    if not inst:
        print("\n  No sequences installed.\n")
        print(MANUAL_INSTRUCTIONS)
    else:
        print(f"\n  Installed sequences ({len(inst)}):\n")
        total_frames = 0
        for s in inst:
            fc = frame_count(s)
            total_frames += fc
            print(f"    ✓  {s:<22}  {fc:>5} frames")
        print(f"\n  Total: {total_frames} frames ready for the app.\n")


def cmd_small():
    print("\n  Installing 2 smallest sequences (~1.6 GB total)…")
    for seq in SEQUENCES:
        if seq[0] in SMALL_SEQS:
            install_sequence(seq)
    print("\n  Done. Run: python endoscope_app.py\n")


def cmd_interactive():
    print("""
╔══════════════════════════════════════════════════════╗
║   C3VD Colonoscopy Dataset Installer                 ║
╚══════════════════════════════════════════════════════╝

  1. Install 2 smallest sequences (~1.6 GB) — recommended
  2. Choose specific sequences to install
  3. List all sequences
  4. Check what is installed
  5. Show manual installation guide
  6. Exit
""")
    choice = input("  Enter choice [1-6]: ").strip()

    if choice == '1':
        cmd_small()
    elif choice == '2':
        cmd_list()
        nums = input("\n  Enter sequence numbers to download (e.g. 1,3,5): ").strip()
        indices = []
        for n in nums.split(','):
            try:
                idx = int(n.strip()) - 1
                if 0 <= idx < len(SEQUENCES):
                    indices.append(idx)
            except ValueError:
                pass
        if not indices:
            print("  No valid numbers entered.")
            return
        total_gb = sum(SEQUENCES[i][2] for i in indices)
        print(f"\n  Will download {len(indices)} sequence(s), ~{total_gb:.1f} GB total.")
        confirm = input("  Continue? [y/N]: ").strip().lower()
        if confirm == 'y':
            for i in indices:
                install_sequence(SEQUENCES[i])
            print("\n  Done. Run: python endoscope_app.py\n")
    elif choice == '3':
        cmd_list()
    elif choice == '4':
        cmd_check()
    elif choice == '5':
        print(MANUAL_INSTRUCTIONS)
    else:
        print("  Exiting.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='C3VD Dataset Installer')
    parser.add_argument('--small',  action='store_true', help='Download 2 smallest sequences')
    parser.add_argument('--list',   action='store_true', help='List available sequences')
    parser.add_argument('--check',  action='store_true', help='Check installed data')
    parser.add_argument('--manual', action='store_true', help='Show manual install guide')
    args = parser.parse_args()

    if args.list:
        cmd_list()
    elif args.check:
        cmd_check()
    elif args.small:
        cmd_small()
    elif args.manual:
        print(MANUAL_INSTRUCTIONS)
    else:
        cmd_interactive()