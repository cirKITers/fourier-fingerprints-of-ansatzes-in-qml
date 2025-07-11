import glob
import os
import yaml
import shutil

experiment_id = input("Experiment ID: ")

mlflow_path = f"./mlruns/{experiment_id}"
backup_dir = "./mlruns_bckp/{experiment_id}"
dry_run = False

cut_after = 0  # set to 0 to disable

if not os.path.exists(backup_dir):
    os.makedirs(backup_dir)

print(f"Searching in: {mlflow_path}")
print(f"Using backup directory: {backup_dir}")

runs = glob.glob(os.path.join(mlflow_path, "*"))
print(f"Found {len(runs)} runs")
print(f"Dry run? {dry_run}")

for r in runs:
    mark_for_deletion = False
    mark_for_deprecation = False
    if not os.path.isdir(r):
        continue
    with open(os.path.join(r, "meta.yaml"), "r") as f:
        try:
            content = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)

        if content["status"] != 3:
            mark_for_deletion = True
        elif int(content["end_time"]) < cut_after:
            mark_for_deprecation = True
            content["lifecycle_stage"] = "deleted"

    if mark_for_deletion:
        print(f"Moving {r} to trash")
        if not dry_run:
            shutil.move(r, os.path.join(backup_dir, os.path.basename(r)))
    elif mark_for_deprecation:
        print(f"Marking {r} as deprecated. Will be deleted next run.")
        if not dry_run:
            with open(os.path.join(r, "meta.yaml"), "w") as f:
                yaml.safe_dump(content, f)
