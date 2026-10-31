# Review the original 206 fixed-gap candidates in native RGB images

Run `bash scripts/review_fixed_gaps.sh`. This resolves the existing current
S_selector run and its unchanged fixed manifest, optionally reuses completed
choice predictions, and exports a CPU-only scene-review atlas. No torch, model
forward, checkpoint loading, optimizer, training, candidate regeneration or
threshold selection is performed. The optional S checkpoint check only hashes
the file to verify which S model initialized choice; missing checkpoint files
do not block RGB/cache review.

Output:

- `overviews/*_marked.png`: all14 full-resolution original images, with each
  fixed missing component highlighted, numbered and linked visually to anchors.
- `originals/*.png`: unmodified RGB pixels, native1500 grid, no resizing.
- `gaps/Gxxxx_rgb.png`: clean RGB context crop; default margin128 native pixels
  around the union of missing component and all anchor groups.
- `gaps/Gxxxx_review.jpg`: aligned RGB, GT/anchors, P0, current S, choice and
  choice-P0 panels. Missing caches are visibly unavailable, never reconstructed
  from screenshots or treated as unchanged predictions. Narrow border crops may
  be padded horizontally for readable captions, but RGB pixels are never resized;
  the separate clean RGB crop and full-image markers retain exact native grids.
- `masks/*_gap_ids.tif`: native uint16 ID raster, G0001=value1, only the exact
  fixed missing pixels receive IDs. Display dilation never changes this raster.
- `gap_locations.json`: all missing/anchor coordinates, crop boxes and original
  component IDs. Coordinates are zero-based native x/y, NOT longitude/latitude.
- `gap_inventory.csv`: native coordinates, source component pixel count/length
  field, number of anchor groups and original-predicate failure details.
- `review_template.csv`: initially blank scene labels; an existing template is
  not overwritten on rerun.
- `index.html`: offline clickable image/candidate review; no CDN/network calls.
  Fields cover road type, vegetation/shadow/buildings/field/parallel-road etc.
  background, scene, anchor reliability, confirmation and uncertainty.
- `summary.json`, `identity.json`, `REPORT.md`: data identity, cache provenance,
  missing inputs and evidence limits.

The script also produces a tar.gz of the complete review directory and prints
`DOWNLOAD_ARCHIVE=...`. Download/extract the whole archive locally and open
`index.html`. Only downloading HTML loses the relative image assets. Browser
annotations are saved best-effort in localStorage; EXPORT JSON/CSV to preserve
work. A previously exported JSON can be imported only with the same manifest
SHA256. Initial labels are unreviewed; no RGB heuristic invents tree/shadow or
road-material classifications. Those categories must be visually confirmed.

Numbered prediction NPZ files are mapped by exact, uniquely matching GT content,
not by their numeric filename order. P0/native-GT/skeleton fingerprint is checked
against the original fixed reference when all P0 caches exist; a mismatch aborts.
An incomplete reference check is explicitly marked as such. Geometry is validated
without image resizing. Source manifest/checkpoints/predictions remain read-only.

Original pass criterion remains >=80% exact missing-coordinate coverage AND all
fixed anchor groups sharing a global predicted component. Failure reasons split
coverage, disappearing anchors and disconnected anchors. Extra 2px-tolerant
coverage and within-context connectivity are diagnostic only; a globally
connected but locally disconnected case may involve a path leaving the context,
not necessarily a wrong connection. These alternatives never replace the old
score. Component pixel count is not endpoint distance; multi-anchor candidates
and uncertain GT should be reviewed rather than called confirmed breaks.

P0 displays at the original0.45. S/choice share the saved common threshold from
choice comparison (or original S's selected threshold when choice is absent).
No validation/test tuning occurs. If the threshold is missing, S/choice scores
are unavailable; RGB/GT/fixed coordinates and available P0 can still be exported.

Five synthetic tests exercise native-coordinate identity masks, raw RGB crop
sizes, reversed numbered-cache mapping, GT mismatch rejection, untouched input
manifest/template, exact original closure, global-only detours, disappearing
anchors, absent-cache behavior and embedded offline-page JSON. Real scene labels
remain unfilled until the server images are reviewed.
