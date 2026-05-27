"""
Metadata-first dataset ingestion from LILA Science NZ Trail Cams.

Workflow:
  1. Download COCO JSON metadata only (~few hundred MB) from LILA S3
  2. Parse into a manifest CSV: image_path, species, site_id, timestamp, binary_label
  3. Apply species-to-binary mapping and per-species caps
  4. Download only the selected ~150-200k images to the DVC remote

Species mapping:
  mammals (invasive predators) → label 1
    mouse, possum, rat, stoat, ferret, feral cat, hedgehog, weasel
  birds (native species) → label 0
    kiwi, kea, tui, weka, fantail, kereru, robin, bellbird, morepork, etc.
  excluded entirely: humans, vehicles, deer, cattle, sheep

Per-species cap: ~8,000 images per mammal species to prevent mouse (49% of
raw dataset) from dominating the training signal. All bird images are retained.
"""
