# Manually generate and replace all 27 synoptic charts

This package is for the existing `professorjake29/Synoptic-Charts` repository on its `main` branch. It has **no automatic schedule**. Use GitHub > Actions > **Generate and Replace Synoptic Charts** > **Run workflow** when you have confirmed the data are available.

## Upload the full package, preserving paths

- `Synoptic_I_Briefing_Charts_GFS_06Z_Forecasts_IEM_ECCC_HYBRID_v6.py` (repository root; replaces its existing same-name script)
- `environment.yml` (root)
- `expected_chart_filenames.txt` (root; explicit list of all 27 existing chart PNG filenames)
- `scripts/publish_charts.py` (new subfolder)
- `.github/workflows/synoptic-charts.yml` (GitHub workflow)
- `README_GITHUB_ACTIONS.md` (optional documentation)

The existing 27 PNG files stay in the root directory. **Do not delete them** when uploading the package.

## How it works

1. You manually trigger the workflow from the Actions tab on branch `main`.
2. GitHub installs the Python environment and runs the cloud-compatible synoptic script. New files go to the temporary `output/` directory.
3. A separate validation step checks that *exactly* the 27 expected filenames exist, and that every file is a readable, sufficiently large PNG.
4. Only after validation succeeds are all 27 images copied over the existing 27 repository-root PNG files.
5. A single git commit replaces the old images. The workflow then pushes the commit to `main`. This means viewers see either the old set or the new set, not a partially committed set.
6. An Actions artifact is also available for download, including partial outputs when a previous step fails.

## Settings

- Repository > Settings > Actions > General > Workflow permissions: **Read and write permissions** may be required. The workflow requests `contents: write`; branch protection/rulesets may also restrict pushes from Actions.
- If a run fails because the branch changed before `git push`, run the workflow again. The workflow refuses to force-push or overwrite concurrent work.
- If your repository uses a default branch other than `main`, update both the `if:` condition and `git push` destination in the workflow.

## Local computer unchanged

Outside of GitHub Actions, the modified Python script still defaults to `/mnt/c/Users/jakewiley/Documents`, clears its PNG files, and then runs `/home/wiley/upload_synoptic_charts_revised_current.sh` at the end. You may also keep your original local script unchanged instead of replacing it on your local machine.

## Important caveats

- This workflow only runs when you click the button. It does **not** inspect UCAR data availability before generating charts; your manual timing controls that.
- The 27-image validation prevents publishing an incomplete or corrupt chart set, but does not verify the meteorological accuracy, timestamps or freshness of a valid-looking chart.
- Full 27-chart rendering has not been tested end-to-end here; a manual GitHub run is needed.
- GitHub still retains prior chart versions in git history, so a large long-term repository may grow over time.
