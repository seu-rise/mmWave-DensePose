# Web GUI

The local workbench provides **Dataset**, **Training & Testing**, **Results**
and **Official results** in an English browser interface. It runs on Windows
and Linux and serves its JavaScript and chart libraries locally. Use a browser
with WebGL2 for 3D views.

The database icons beside dataset and official asset downloads open their
ScienceDB publication page. Question-mark icons offer short context help on
hover or keyboard focus; click to keep a hint open and press Escape to dismiss.

## Start

Run commands from the repository root. For dataset browsing, saved logs and
official figures, the base environment is sufficient:

```bash
uv sync --locked
uv run --locked python app.py --data-root data/mmwave_densepose --trust-pickle --open-browser
```

Only enable `--trust-pickle` for trusted dataset and prediction files. Loading a
pickle can execute code on the server. Without this flag, numeric NPY/NPZ data
can still be browsed, but PKL files cannot be loaded.

To launch training or testing from the GUI, start the service with a compute
profile. For NVIDIA CUDA 11.8:

```bash
uv sync --locked --extra cu118
uv run --locked --extra cu118 python app.py --data-root data/mmwave_densepose --trust-pickle --open-browser
```

Use `cpu` instead of `cu118` for CPU execution. Keep the same selected profile
when synchronizing an environment that already contains training dependencies.
See [environment setup](./environment.md).

Open the complete URL printed by the service, including `#token=...`, and keep
it private. The default address is `http://127.0.0.1:8765/`. Each restart creates
a new token; use the newly printed URL if an old page reports an expired token.
Stop the service with Ctrl+C. Add `--port 8766` to run a separate instance.
After updating Python code, restart the service when its tasks have finished.

## Dataset

The sequence selector reads files from the **server's** `--data-root`, not the
browser computer. Select a sequence to load it; there is no upload step. Restart
with another `--data-root` to browse a different directory. One service shares
one loaded sequence across its clients; use separate ports for independent
sessions.

**Overview** displays frozen release statistics, not a live count of the local
files. The action beside Overview depends on the installation:

- **Download dataset** starts package download and installation when data is
  absent and published package links are available.
- **Verify dataset** explicitly scans installed files against the release
  baseline. Opening Overview alone does not scan the dataset.
- After an unsuccessful verification, inspect the affected items and suggested
  repairs. Package repair downloads the required parts rather than the whole
  dataset.

The progress dialog reports downloading, extraction, installation and
verification. Close it to keep browsing; reopen the action to inspect progress.
Use its cancellation control to cancel an operation.

**Download availability:** the dataset is released. Download and repair use the
package URLs and checksums in the release manifest. See
[dataset installation](./dataset.md) for the installation workflow.

The viewer supports complete sequence PKL/pickle files, numeric sequence NPZ
files and NPY arrays shaped `[frames, points, channels]`. Split/index files are
not sequences. Image channels and skeletons appear only when the selected file
contains them.

| Control | Action |
| --- | --- |
| Sequence and adjacent arrows | Load a sequence or move to the previous/next one |
| Timeline, frame number and step buttons | Navigate zero-based frame indices |
| Play/Pause | Play at a target 6.5 fps; decoding or transfer may reduce speed |
| Radar / Skeleton / Boundary | Show or hide scene layers |
| Color / Size | Choose point coloring and visual point scale |
| Drag / mouse wheel / middle-drag | Orbit / zoom / pan |
| Reset / Fit | Restore the view / fit the current frame |

RGB, depth, infrared and body-index images follow the stored frame association.
Missing channels remain missing. The viewer does not recalibrate or anonymize
the source files. Use only the anonymized release for public image display.

## Training & Testing

### Execution

The Task selector contains **Train**, **Test** and **Predict**. Standalone
validation remains available through the [CLI](./training.md); training still
performs its configured validation phase.

For **Train**, choose the model and main parameters. **Other parameters** opens
the grouped full configuration. Parameters that do not apply to the selected
model or task are hidden. Use the GPU checkboxes for CUDA training; selecting
multiple GPUs uses a total batch size shared across those GPUs.

For **Test** or **Predict**, select **Weights / Training run** in the main
configuration panel. This restores the selected weight's recorded model, input
features, point count, temporal window, seed, preprocessing and split settings.
These settings are locked to preserve the evaluation protocol. Device, one
selected GPU, batch size, loader/runtime settings and supported output options
remain editable. Missing official weights can be selected to view download
status and install the corresponding package. **Download official weights**
shows progress and cancellation; unavailable package links are displayed with a
reason. Evaluation remains disabled until the weight passes compatibility
checks. To use a different protocol, select a weight trained with it. See
[official asset downloads](paper-assets.md) for CLI and offline installation.

Enable **Visual · Prediction PKL** before a pose Test if you need frame-by-frame
point-cloud, skeleton and avatar playback in Results. A metrics-only test does
not provide that playback. Predict exports predictions for its configured
labeled split.

Click **Run** to validate. An invalid configuration opens All parameters with
every error and warning, including issues for hidden settings. Select an issue
to locate its group and field. Diagnostics provide the cause, suggested repair,
current/expected values where available, and expandable error details. A weight
mismatch can offer **Restore selected weight settings**, which reloads the
recorded settings and validates again without starting a task.

When validation succeeds, review the effective parameters and output paths,
then choose **Start run**. The active-run card shows progress, metrics and
elapsed time; Console shows process output. **Parameters** opens the selected
run's read-only settings. **Stop** is enabled only for a running process the
service can identify and control.

The right-side shortcut opens **Logs** for training or **Results** for a
completed Test. The Results shortcut selects that exact model, training run and
test ID; it stays disabled until the test finishes. Local CLI tasks launched
from this repository also appear in Execution with a **CLI** label.

### Logs

Logs lists **training runs only**. Choose **My training** or **Official training**
using the tabs above Model, filter by model, select runs, and
open a run's details icon to inspect its recorded parameters and artifacts.
Use each plot's metric selector to choose what it displays. Hover over a curve
for recorded values; use the axis and smoothing controls to inspect training.
Each plot leaves space after its last point. Hover in that space to read the
final value of each selected curve. The translucent tooltip keeps curves visible.

**Native TensorBoard** opens in a new browser tab and loads all indexed training
logs that contain TensorBoard event files, regardless of the current filters or
selection. Runs are named by origin, model and training run so they can be
distinguished in TensorBoard. There is no intermediate run-selection dialog.

Deleting personal training logs moves them to local trash. Their finished task
records are retired too; model checkpoints and test results are retained.
Official logs and active tasks are protected.

## Results

Choose **My evaluation** or **Official evaluation**, then select **Model →
Weights → Test ID**. Running a local test with an official weight creates a
record under My evaluation. It does not replace the official evaluation.
Official table tags beside weight/test options identify their associations.
When the selected official test needs prediction files, its visualization area
shows the prediction package status and a download button. Installation runs in
the background while metrics remain visible, then refreshes the selected record.
Tests without a published prediction package remain metrics-only.

The panel shows the saved test metrics and per-joint errors. When available,
the 19-joint diagram highlights a joint on hover and shows its error in
centimeters. A usable prediction PKL additionally enables frame playback,
predicted/ground-truth poses, radar points and the Xbot avatar. The service must
have been started with `--trust-pickle` to load that PKL. The avatar is off by
default and its mesh loads only after enabling **Avatar**. Its checkbox, model,
pose source and opacity controls are grouped together below the visualization.

Use **Compare runs** to compare saved metrics as tables and plots. Spatial
sample comparisons require compatible records and matching sample IDs.
Comparison does not rerun inference or change the recorded evaluation threshold.

**Delete test** is available for personal evaluations. Its confirmation identifies
the model, training run and test ID. It moves only that test's logs and prediction
artifacts to local trash; other tests, training logs and checkpoints remain.
Official and active records cannot be deleted. **Refresh** updates the inventory.

## Official results

Figures and the **Official** table source use frozen reference data. Tables
1–3 are informational, so **My results** is disabled. For supported experiment
tables, My results aggregates available complete local tests that match the
table's model and protocol; missing results display a dash.

Expandable subplots highlight their border on hover or keyboard focus. Click
the whole subplot, or press Enter/Space, to open its interactive dialog. The
main Three.js/ECharts previews do not rotate or zoom. The per-joint comparison
keeps inline value tooltips; expanded charts offer their interactive controls.
Captions appear below the figure or table. See
[official results and reproduction](./official-results.md).

## Remote use and troubleshooting

For a headless server, omit `--open-browser` and keep the service on loopback.
From your browser computer, forward the service port:

```bash
ssh -N -L 8765:127.0.0.1:8765 user@server
```

Open the server's complete token-bearing URL locally. If changing ports, use
the same port in `--port` and both sides of the tunnel. Keep the tunnel open.
Use SSH forwarding instead of exposing the service to the public Internet.

| Problem | Action |
| --- | --- |
| Invalid/expired token | Open the complete URL printed by the current server |
| Old behavior after an update | Finish active tasks, restart the Python service, then reload its new URL |
| No sequences | Check the server's data directory and installation status |
| PKL refused | Confirm the file's source, then restart with `--trust-pickle` if appropriate |
| Training unavailable or CUDA missing | Start with the required compute profile, or choose CPU |
| No result playback | Check that this test generated a prediction PKL with Visual enabled |
| Slow loading | Wait for the loading/progress dialog; large multimodal files require substantial memory |

Return to the [repository README](../../README.md).
