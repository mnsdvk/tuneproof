# Deploying VisionPilot on AWS (Graviton + COOL)

Everything below must be run in **your** AWS account; nothing here was deployed from the build sandbox.

## Option A: single Graviton EC2 instance (fastest, ~20 min)
1. Subscribe to **OpenCV 5 for Graviton4** (or Graviton5) on AWS Marketplace; launch from that AMI on a
   `c8g.xlarge` (Graviton4) or matching Graviton5 type. Security group: allow 8080 from your IP only (or put behind an ALB + HTTPS).
2. `git clone` / `scp` this repo to the instance, then: `./deploy/run_on_graviton.sh cool-graviton4`
3. Start the endpoint: `nohup python3 -m visionpilot.server --port 8080 > server.log 2>&1 &`
4. Check `curl localhost:8080/api/health` shows `"machine": "aarch64"` and the COOL OpenCV build.
5. Ship logs to CloudWatch (access log is single-line JSON; install the CloudWatch agent and tail `server.log`).

## Option B: container on ECS/EKS (Graviton nodes)
Build with the COOL base image (see Dockerfile header), push to ECR, run as a Fargate ARM64 task / EKS pod.
The COOL container is billed per task/pod-hour through the Marketplace entitlement.

## Hybrid x86/Arm (optional, strengthens the COOL story)
Run the same container on an x86 instance as the baseline arm and compare cost per 1,000 frames using `bench.py`
numbers and the instance on-demand prices you look up on the day. Document both in the report.

## Security notes
- No credentials in code; instance role only. Endpoint accepts max 20 MB bodies, images only, no file paths from clients.
- Approvals are appended to `approvals.jsonl` (set `VP_APPROVALS`); contains bounding boxes and decisions only.
