# MediBot hospital delivery simulation

MediBot is a code-generated Webots world and controller for one simulated medicine delivery at a time. The world, hospital, AprilTags, visitor models, and robot are generated from Python source.

## Windows setup

Use a Command Prompt or PowerShell opened in the repository folder. The `py` launcher selects the installed Python used for project setup and checks:

```powershell
py --version
py -m pip install -r requirements.txt
py generate_world.py
py tests\test_core.py
py tests\test_handover.py
py tests\test_controller_import.py
```

Open `worlds\medibot.wbt` with Webots using **File > Open World...**, then press **Run**. In Webots, set **Tools > Preferences > Python command** to the same Python installation that has the dependencies. Webots does not necessarily use the Python selected by `py` automatically.

**Do not press Save in Webots.** The world is generated code and a Webots save can be overwritten. To restore generated files, close or revert the world and run `reset_world.bat` from the repository folder. Edit `generate_world.py` or the controller/config source, then regenerate; do not hand-edit `worlds\medibot.wbt`.

## Keys

Click inside the 3D view to give it keyboard focus. Tap and release keys individually; the obstacle and visitor controls are debounced.

| Key | Action |
|---|---|
| `S` | Start one delivery to the default destination, Ward 3B |
| `B` | Place the next obstacle on the active route; during a mission use it after the robot reaches the JUNCTION-to-WARD-2A corridor |
| `U` | Remove spawned obstacles and clear temporary blocked edges |
| `V` | Select the next visitor: patient's relative, another patient's relative, then an unauthorized visitor |
| `C` | Swap to another person's card; press again to restore the selected visitor's card |
| `T` | Toggle AprilTag pose correction while IDLE |
| `X` | Reset the robot to HOME and restore the default visitor/card selection |
| `F`, `R`, `1`, `2`, `3` | Development bypass keys; only enabled when `controllers\medibot_controller\dev_mode.flag` exists, with use logged as `DEV_OVERRIDE` |

The normal `V` and `C` demo controls only select the presented face/card. The normal patient-specific face and RFID authorization checks still decide whether the compartment can open.

## Scenario recipes

Start each recipe from IDLE. After a mission completes, press `X` before the next recipe. Expected console lines below are examples of the controller's event messages; exact IDs, timing, and route node lists depend on the selected patient and simulation run.

### A. Authorized delivery

1. Press `X` once if needed to reset, then press `S`.
2. Leave the default visitor and card selected; wait for the robot to return and dock.

Expected console messages include:

```text
[A*] route: HOME -> ... -> WARD-3B-IN
[HANDOVER] FACE: STUB id=... score=1.00
[RFID] UID=... read
[HANDOVER] AUTHORIZED patient=... compartment=...
[LOCK] compartment ... UNLOCKED
[MISSION LOG] DELIVERED appended to ...\logs\deliveries.csv
```

### B. Obstacle and reroute

1. Press `X`, then `S`.
2. Wait until the console prints `[NAV] reached JUNCTION -> heading to WARD-2A`, then tap `B` once and release it.
3. Wait for the alternate route, delivery, and return to HOME.

Expected messages include:

```text
[TROLLEY] placed at (x,y), d=1.00 m ahead
[OBSTACLE] STOP ...
[REPLAN] corridor ... blocked ...
[A*] route: JUNCTION -> ICU -> ...
[HANDOVER] AUTHORIZED ...
[MISSION LOG] DELIVERED appended to ...\logs\deliveries.csv
```

### C. Unauthorized visitor

1. Press `X`, then tap and release `V` twice while IDLE. This selects an unauthorized visitor with that visitor's own card.
2. Press `S`. Let the verification and return finish.

Expected messages include `[HANDOVER] DENIED attempt=...: face not authorised for this patient` and a final `[MISSION LOG] DENIED appended to ...\logs\deliveries.csv`. The assigned lid should not print an `UNLOCKED` event for this handover.

### D. AprilTag correction OFF versus ON

1. From a reset, tap `T` once while IDLE to turn correction **OFF**, then press `S` and let the run finish.
2. Press `X`, tap `T` once to turn correction **ON**, then press `S` again.
3. Run `py plot_log.py` to compare the logged error and driven/planned paths.

The console should print `[TAGS] AprilTag correction OFF` and later `[TAGS] AprilTag correction ON`. The controller writes per-step logs in `controllers\medibot_controller\logs` with a `tags_on` column. The size of the drift is a simulation result and may vary between runs.

## What is simulated

- Webots runs the physics and simulated wheel, IMU, camera, and ultrasonic sensors. The hospital, furniture, robot, and visitors are 3D models.
- Visitor avatars display generated, stylized face illustrations. Face recognition is currently a clearly marked **STUB**: it uses the Supervisor-selected visitor identity and returns score `1.0`; it does not recognize a face from image pixels.
- RFID is emulated in the controller. It accepts a card when the Supervisor-computed card position enters the reader's distance and angle zone; this is not a physical RFID radio simulation.
- `V` and `C` select which visitor and card are presented. They do not bypass authorization. Development bypass keys require `dev_mode.flag` and write `DEV_OVERRIDE` outcomes.
- Pose-error values and plots compare the simulated estimate with Webots' simulated Supervisor pose. They are simulation results, not measurements from a physical robot.

## Generated viewpoints and recording

The generated world includes `VIEW_OVERVIEW` (the default overhead hospital view) and `VIEW_FOLLOW_ROBOT` (a tracking shot following `MediBot`). Choose the viewpoint from the Webots viewpoint controls/scene tree; the follow preset uses Webots' `follow` and `followType` fields.

For a demo video, open **Tools > Movie recording**, set the recording resolution to **1280 x 720**, choose an output file, and start recording before pressing `S`. Stop recording after the robot docks. For a live camera image, use **Overlays > Camera Devices > camera**.

## Logs and plots

- `controllers\medibot_controller\logs\deliveries.csv` appends one summary row for each mission outcome.
- `mission_*.csv` records estimated and simulated true pose, error, tag-correction state, and the planned waypoint target over time.
- `tag_corrections_*.csv` records tag detections and correction decisions.
- Run `py plot_log.py` to write `localisation_comparison.png` at 200 dpi. Its two panels show error over time (tags ON/OFF) and the driven path against recorded planned waypoints.

## Troubleshooting

| Symptom | What to check |
|---|---|
| Webots reports missing `numpy`, `cv2`, or another Python import | In **Tools > Preferences > Python command**, select the same Python installation you used for `py -m pip install -r requirements.txt`. If Webots' bundled Python has a different version, install the dependencies into that exact interpreter or change the Webots Python command to your `py` installation. |
| Visitor face is blank or a face texture cannot load | Run `py generate_world.py` to regenerate `worlds\textures\faces`. The face decision is still the STUB; this project does not require or load a trained face-model file. |
| AprilTag corrections do not appear | Confirm the camera overlay shows wall tags and that the console reports tags enabled. Press `T` only while IDLE to turn corrections back ON. |
| `B` does not place an obstacle | Use it after the robot reaches the JUNCTION-to-WARD-2A corridor; earlier edges may not have a safe alternate path or the 1 m placement may be too close to a node. |
| Generated world changes disappear | Do not save the world from Webots. Run `reset_world.bat` to regenerate the `.wbt` and generated assets. |
| Webots reports a world parse/controller error | Copy the first Webots error and controller traceback from the console; then regenerate with `py generate_world.py` and check that Webots opened this repository's `worlds\medibot.wbt`. |

## Before the demo

1. Open a terminal in the repository and check `py --version`.
2. Run `py -m pip install -r requirements.txt` if this is a new machine/environment.
3. Run `py generate_world.py` and confirm it reports validation passed.
4. Run `py tests\test_core.py`, `py tests\test_handover.py`, and `py tests\test_controller_import.py`.
5. Set Webots' Python command to the interpreter with the project dependencies.
6. Open `worlds\medibot.wbt`; do not press Save.
7. Confirm the console prints `[MediBot] ready` and the dashboard is visible.
8. Test one authorized mission with `S`, then reset with `X`.
9. Record the demo at 1280 x 720; for the obstacle scenario press `S`, wait for JUNCTION toward WARD-2A, then tap `B`.
10. After the run, check `logs\deliveries.csv` and generate the comparison with `py plot_log.py` if showing the drift demo.
