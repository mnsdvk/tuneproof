#!/usr/bin/env bash
# Run on the Graviton COOL AMI. Produces all CoolPilot evidence in ./results and ./evidence.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results evidence
uname -m | tee evidence/arch.txt
TOKEN=$(curl -s -X PUT -H 'X-aws-ec2-metadata-token-ttl-seconds: 60' http://169.254.169.254/latest/api/token || true)
curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-type | tee evidence/instance_type.txt; echo
python3 -c "import cv2;print(cv2.__version__, cv2.__file__);print(cv2.getBuildInformation())" > evidence/opencv_build.txt
head -3 evidence/opencv_build.txt
grep -i -E "kleidi|neon|sve|armpl" evidence/opencv_build.txt | head -10 || true
python3 -m unittest discover -s tests
python3 -m coolpilot.run --label "${1:-cool-graviton5}" --out results/coolpilot_cool.json --images 8 --repeats 5 --grid
echo
echo "NEXT: stock-pip comparison on this SAME instance:"
echo "  python3 -m venv /tmp/pipenv && /tmp/pipenv/bin/pip install -r requirements.txt"
echo "  /tmp/pipenv/bin/python -m coolpilot.run --label pip-graviton --out results/coolpilot_pip.json --images 8 --repeats 5 --grid"
echo "  python3 -m coolpilot.run --compare results/coolpilot_pip.json results/coolpilot_cool.json"
