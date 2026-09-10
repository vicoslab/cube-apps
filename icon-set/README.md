# ViCoS demo app icons

Original, editable SVG artwork matching the existing white-on-coral menu. No generated raster images, external fonts, linked assets, or runtime dependencies.

## Style

- 64 × 64 viewBox; transparent background; explicit white strokes.
- 2.2-unit main strokes, round joins/caps; optically balanced margins.
- STOP lettering uses DejaVu Sans Condensed Bold converted to filled SVG outlines (no runtime font dependency), centered with generous border clearance.
- Designed for the GUI's single-channel icon mask; no color-dependent details.
- Marine-polyp icon depicts a sessile polyp with tentacles, not a medical polyp.

## Installed assets

| Master | Existing destination |
|---|---|
| cloth-gripping.svg | NiryoClothDemo/icon.svg |
| wood-classification.svg | BoardDemo/icon.svg |
| road-signs.svg | TrafficDemo/icon.svg |
| defect-detection.svg | PlosciceSupervisedDemo/icon.svg |
| object-counting.svg | CountingDemo/icon.svg and CountingDemo-GeCo2/icon.svg |
| marine-polyps.svg | PolypDemo/icon.svg |
| default.svg | ../demo-default-icon.svg |

These are copied assets, not symlinks. When editing a master, update the corresponding installed copy too. Existing cfg.xml paths and all application behavior are unchanged. Restart the GUI to reload cached icon textures.

## Additional assets

- sketch-recognition.svg and object-tracking.svg cover apps shown in the reference screenshot but absent from this checkout. Copy each into its app as icon.svg when that app is available.
- cloth-corners.svg is available for the disabled ClothDemo. Its disabled configuration was deliberately left unchanged; it still references the generic fallback, as does PlosciceDemo.

## Verification

All ten masters parsed as XML and rasterized using MuPDF at 32, 64, 100, and 256 px, with nonempty, unclipped alpha bounds. Installed app copies were byte-compared with their masters; all six enabled app configurations resolve to valid SVG assets. The GUI/hardware stack was not launched. The menu preview is a composite of the supplied screenshot with replacement icons, not a live screenshot.
