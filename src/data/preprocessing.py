"""
Image preprocessing and augmentation for camera trap images.

Augmentation policy (torchvision.transforms.v2 + Albumentations):
  INCLUDED:
    - Motion blur (simulates animal movement)
    - Gaussian noise (low-light sensor noise)
    - Brightness/contrast jitter (lighting variation across sites)
    - Coarse dropout (vegetation occlusion)
    - Horizontal flip

  EXCLUDED:
    - Colour jitter / saturation on IR night images (greyscale — misleading)
    - Vertical flip (cameras are fixed, gravity is constant)
    - Heavy crop (risks removing animal entirely from small frames)

Seasonal-aware config: winter raises noise strength and lowers brightness
floor; summer raises coarse dropout weight to simulate dense NZ foliage.
Config is selected per-image from the timestamp in COCO metadata.

Output: 224×224 tensors normalised with ImageNet mean/std.
"""
