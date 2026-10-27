#!/usr/bin/env bash
# Run on a Graviton EC2 instance (e.g. c8g.xlarge) launched from the COOL AMI
# (AWS Marketplace: "OpenCV 5 for Graviton4"/"Graviton5"). Produces the evidence
# the Best Use of COOL award asks for: build info, tests, accuracy, and timings.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results evidence
uname -m | tee evidence/arch.txt
curl -s -H "X-aws-ec2-metadata-token: $(curl -s -X PUT -H 'X-aws-ec2-metadata-token-ttl-seconds: 60' http://169.254.169.254/latest/api/token)" \
  http://169.254.169.254/latest/meta-data/instance-type | tee evidence/instance_type.txt; echo
python3 -c "import cv2;print(cv2.__version__);print(cv2.getBuildInformation())" > evidence/opencv_build.txt
grep -i -E "kleidi|neon|sve|armpl|version" evidence/opencv_build.txt | head -20
python3 -m unittest discover -s tests
python3 -m visionpilot.evaluate --n 100 --seed 7 --out results/eval_graviton.json
python3 -m visionpilot.bench --label "${1:-cool-graviton}" --runs 50 --out "results/bench_${1:-cool}.json"
echo "Now repeat the bench in a clean venv with stock pip OpenCV on the SAME instance type:"
echo "  python3 -m venv /tmp/pipenv && /tmp/pipenv/bin/pip install -r requirements.txt"
echo "  /tmp/pipenv/bin/python -m visionpilot.bench --label pip-graviton --runs 50 --out results/bench_pip.json"
echo "  python3 -m visionpilot.bench --compare results/bench_pip.json results/bench_${1:-cool}.json"
