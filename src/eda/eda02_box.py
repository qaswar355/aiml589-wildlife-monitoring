"""
EDA Step 2 — Bounding boxes for the XAI evaluation set.
 
Pulls two things:
 
  A. MegaDetector v5a + RDE results for nz-trailcams.
     These are MODEL PREDICTIONS. Usable as pseudo-ground-truth for a
     pointing-game evaluation, but they are not human annotations.
 
  B. mdv5_lila_boxes.zip — the ~1.1M HUMAN-ANNOTATED boxes used for MDv5
     training, across all of LILA. MDv5b was trained on "a small subset of
     Trail Camera Images of New Zealand Animals", so some of our images may
     have REAL boxes in here. If so, that subset is a much stronger XAI
     evaluation set.
 
Usage:
    pip install ijson requests pandas pyarrow
    python eda_02_boxes.py            # downloads to data/boxes/, then reports
 
Outputs:
    data/md_boxes.parquet   image_id, n_dets, max_conf, bbox (highest-conf animal)
    data/gt_boxes.parquet   image_id, bbox_relative   (only if NZ images found)
"""
 
import json
import sys
import zipfile
from pathlib import Path
 
import pandas as pd
import requests
 
DATA = Path("data")
BOXES = DATA / "boxes"
BOXES.mkdir(parents=True, exist_ok=True)
 
MD_RDE_URL = ("https://lila.science/public/lila-md-results/"
              "nz-trailcams_mdv5a.0.0_results.filtered_rde_0.150_0.850_40_0.200.zip")
GT_URL = "https://lila.science/public/md-splits/mdv5_lila_boxes.zip"
 
FLAT = DATA / "nz_flat.parquet"
CONF_THRESHOLD = 0.2   # MD's own recommended floor; we report across thresholds anyway
 
 
# ---------------------------------------------------------------------------
def download(url: str, dest: Path) -> Path:
    if dest.exists():
        print(f"  [cached] {dest.name} ({dest.stat().st_size / 1e6:.0f} MB)")
        return dest
    print(f"  downloading {dest.name} ...", flush=True)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with dest.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r    {done/1e6:6.0f} / {total/1e6:.0f} MB", end="", flush=True)
    print()
    return dest
 
 
def unzip_one(zpath: Path) -> Path:
    """Extract the single .json inside a LILA results zip."""
    with zipfile.ZipFile(zpath) as z:
        names = [n for n in z.namelist() if n.endswith(".json")]
        if not names:
            raise RuntimeError(f"no .json inside {zpath.name}: {z.namelist()}")
        if len(names) > 1:
            print(f"  note: {len(names)} json files inside; using {names[0]}")
        out = BOXES / Path(names[0]).name
        if not out.exists():
            print(f"  extracting {names[0]} ...", flush=True)
            with z.open(names[0]) as src, out.open("wb") as dst:
                while chunk := src.read(1 << 20):
                    dst.write(chunk)
    return out
 
 
# ---------------------------------------------------------------------------
def parse_md_results(jpath: Path) -> pd.DataFrame:
    """
    Stream the MD results file. It covers 2.45M images, so json.load() would
    want ~10GB+ of RAM. ijson keeps it flat.
 
    Expected shape:
      {"images":[{"file": "...", "detections":[{"category":"1","conf":0.9,
                                                "bbox":[x,y,w,h]}]}],
       "detection_categories": {"1":"animal","2":"person","3":"vehicle"}}
    bbox is [x_min, y_min, width, height], normalised to [0,1].
    """
    import ijson
 
    # Peek at the non-images keys first — never assume the schema.
    with jpath.open("rb") as f:
        head = f.read(4000).decode("utf-8", "replace")
    print(f"\n  file head:\n    {head[:400]}...\n")
 
    rows = []
    animal_cat = None
    with jpath.open("rb") as f:
        try:
            cats = next(ijson.items(f, "detection_categories"))
            animal_cat = next((k for k, v in cats.items() if v == "animal"), "1")
            print(f"  detection_categories: {cats}  → animal = '{animal_cat}'")
        except StopIteration:
            animal_cat = "1"
            print("  !! no detection_categories found; assuming animal = '1'")
 
    with jpath.open("rb") as f:
        for i, img in enumerate(ijson.items(f, "images.item")):
            dets = [d for d in (img.get("detections") or [])
                    if str(d.get("category")) == animal_cat]
            if dets:
                best = max(dets, key=lambda d: float(d["conf"]))
                rows.append((img["file"], len(dets),
                             float(best["conf"]), [float(x) for x in best["bbox"]]))
            else:
                rows.append((img["file"], 0, 0.0, None))
            if i and i % 250_000 == 0:
                print(f"    {i:,} images parsed ...", flush=True)
 
    df = pd.DataFrame(rows, columns=["md_file", "n_dets", "max_conf", "bbox"])
    print(f"  parsed {len(df):,} image records")
    return df
 
 
def parse_gt_boxes(jpath: Path) -> pd.DataFrame:
    """The human-annotated MDv5 training boxes. COCO-ish, with `bbox_relative`."""
    print("\n  loading GT boxes (this one fits in RAM) ...", flush=True)
    with jpath.open() as f:
        coco = json.load(f)
    print(f"  top-level keys: {sorted(coco.keys())}")
    imgs = pd.DataFrame(coco["images"])
    anns = pd.DataFrame(coco["annotations"])
    print(f"  images: {len(imgs):,}   annotations: {len(anns):,}")
    print(f"  image cols: {list(imgs.columns)}")
    print(f"  annot cols: {list(anns.columns)}")
    if len(imgs):
        print(f"  sample file_name: {imgs.iloc[0].get('file_name')}")
    return imgs, anns
 
 
# ---------------------------------------------------------------------------
def align(md_file: str, our_ids: set) -> str | None:
    """
    MD paths are 'relative to the same base folder as the .json metadata'.
    Our image_id is e.g. 'ACC/banded_rail/UUID.JPG'. Should match directly,
    but strip a leading 'nz-trailcams/' if the unified prefix is present.
    """
    for cand in (md_file, md_file.removeprefix("nz-trailcams/")):
        if cand in our_ids:
            return cand
    return None
 
 
def main() -> None:
    if not FLAT.exists():
        sys.exit(f"missing {FLAT} — run eda01 first")
 
    flat = pd.read_parquet(FLAT, columns=["image_id", "category"])
    our_ids = set(flat["image_id"])
    print(f"manifest: {len(our_ids):,} images\n")
 
    # ---- A. MegaDetector predictions --------------------------------------
    print("=" * 66)
    print("A. MEGADETECTOR RDE RESULTS (predictions — pseudo-ground-truth)")
    print("=" * 66)
    md_json = unzip_one(download(MD_RDE_URL, BOXES / "nz_md_rde.zip"))
    md = parse_md_results(md_json)
 
    md["image_id"] = md["md_file"].map(lambda p: align(p, our_ids))
    matched = md["image_id"].notna()
    print(f"\n  path alignment: {matched.sum():,} / {len(md):,} matched our manifest")
    if matched.sum() == 0:
        print("  !! ZERO MATCHES — path convention differs. Sample MD path:")
        print(f"     {md['md_file'].iloc[0]}")
        print(f"     vs our image_id: {next(iter(our_ids))}")
        sys.exit("fix the path join before continuing")
 
    md = md[matched].copy()
    md[["image_id", "n_dets", "max_conf", "bbox"]].to_parquet(DATA / "md_boxes.parquet",
                                                              index=False)
 
    print("\n  COVERAGE — images with >=1 animal box above threshold:")
    for t in (0.1, 0.2, 0.5, 0.8, 0.9):
        n = int((md["max_conf"] >= t).sum())
        print(f"    conf >= {t:.1f}   {n:>9,}  ({100*n/len(our_ids):5.1f}% of manifest)")
 
    # Does coverage differ by class? A predator/bird gap would bias XAI eval.
    j = flat.merge(md, on="image_id", how="left")
    j["has_box"] = j["max_conf"].fillna(0) >= CONF_THRESHOLD
    print(f"\n  coverage at conf>={CONF_THRESHOLD} for the classes we care about:")
    for sp in ["mouse", "rat", "stoat", "possum", "cat", "kiwi", "kea", "robin", "tomtit"]:
        s = j[j["category"] == sp]
        if len(s):
            print(f"    {sp:<10}{s['has_box'].mean():>6.1%}  of {len(s):>9,}")
 
    # ---- B. Real human boxes ----------------------------------------------
    print("\n" + "=" * 66)
    print("B. HUMAN-ANNOTATED MDv5 BOXES — are any of OUR images in here?")
    print("=" * 66)
    gt_json = unzip_one(download(GT_URL, BOXES / "mdv5_lila_boxes.zip"))
    gimgs, ganns = parse_gt_boxes(gt_json)
 
    fn = gimgs["file_name"].astype(str)
    nz = gimgs[fn.str.contains("nz-trailcams", case=False, na=False)].copy()
    print(f"\n  >>> images whose path mentions nz-trailcams: {len(nz):,}")
 
    if len(nz) == 0:
        print("\n  No real boxes for this dataset. XAI eval uses MD predictions")
        print("  as pseudo-ground-truth — still quantitative, but describe it")
        print("  honestly as such in the methodology.")
        return
 
    nz["image_id"] = nz["file_name"].map(
        lambda p: align(str(p).split("nz-trailcams/", 1)[-1], our_ids))
    hit = nz["image_id"].notna()
    print(f"  of those, matched to our manifest: {hit.sum():,}")
 
    if hit.sum():
        g = ganns.merge(nz[hit][["id", "image_id"]],
                        left_on="image_id", right_on="id", how="inner",
                        suffixes=("_ann", "_img"))
        out = nz[hit][["image_id", "file_name"]].copy()
        if "split" in nz.columns:
            print(f"\n  split distribution: {nz[hit]['split'].value_counts().to_dict()}")
        out.to_parquet(DATA / "gt_boxes.parquet", index=False)
 
        lab = flat[flat["image_id"].isin(set(out["image_id"]))]
        print(f"\n  >>> {len(out):,} images with REAL human boxes.")
        print("  class breakdown of that GT subset:")
        print(lab["category"].value_counts().head(15).to_string())
        print(f"\n  Wrote {DATA/'gt_boxes.parquet'}")
        print("  This is your quantitative XAI evaluation set.")
 
 
if __name__ == "__main__":
    main()