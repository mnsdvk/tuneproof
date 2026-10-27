#!/usr/bin/env bash
# Exact steps used for the AWS run (Ubuntu 24.04 EC2). Run from the unzipped project folder.
set -euo pipefail
sudo apt-get update -y && sudo apt-get install -y python3-venv unzip
python3 -m venv ~/v && ~/v/bin/pip install -r requirements.txt
~/v/bin/python -m unittest discover -s tests
uname -m; ~/v/bin/python -c "import cv2;print(cv2.__version__)"
~/v/bin/python -m coolpilot.run --label aws-stock --out results/coolpilot_aws.json --images 6 --repeats 3 --grid
