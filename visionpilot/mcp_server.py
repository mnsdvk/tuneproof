"""Minimal MCP server (stdio, JSON-RPC 2.0, no SDK needed) exposing the OpenCV 5
tools so *any* MCP-capable agent (Claude, Kiro, Codex...) can drive the same
perceive -> decide -> act loop with its own planner.

    python -m visionpilot.mcp_server
"""
from __future__ import annotations

import json
import sys

import cv2
import numpy as np

from . import service, synth, tools
from .agent import run_agent

_STATE: dict = {"img": None}

TOOL_DEFS = [
    {"name": "load_sample", "description": "Load a synthetic frame (clean|blur|dark|noise|mixed).",
     "inputSchema": {"type": "object", "properties": {"mode": {"type": "string"}, "seed": {"type": "integer"}}}},
    {"name": "load_image", "description": "Load an image file from disk as the working frame.",
     "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
    {"name": "assess_quality", "description": "Blur/exposure/noise evidence for the working frame.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "detect_defects", "description": "Detect defect candidates with scores.",
     "inputSchema": {"type": "object", "properties": {"sensitivity": {"type": "number"}}}},
    {"name": "apply_repair", "description": "Apply fix_exposure | denoise | sharpen to the working frame.",
     "inputSchema": {"type": "object", "properties": {"op": {"type": "string"}}, "required": ["op"]}},
    {"name": "run_full_inspection", "description": "Run the built-in agent loop end to end; returns the decision trace.",
     "inputSchema": {"type": "object", "properties": {}}},
]


def call(name, args):
    if name == "load_sample":
        ds = synth.make_dataset(50, 1000 + int(args.get("seed", 0)))
        pick = [d for d in ds if d["mode"] == args.get("mode", "mixed")] or ds
        _STATE["img"] = pick[0]["image"]
        return {"loaded": True, "shape": list(_STATE["img"].shape)}
    if name == "load_image":
        img = cv2.imread(args["path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("cannot read image")
        _STATE["img"] = img
        return {"loaded": True, "shape": list(img.shape)}
    if _STATE["img"] is None:
        raise ValueError("no frame loaded; call load_sample or load_image first")
    if name == "assess_quality":
        return tools.assess_quality(_STATE["img"])[1]
    if name == "detect_defects":
        return tools.detect_defects(_STATE["img"], float(args.get("sensitivity", 1.0)))[1]
    if name == "apply_repair":
        op = args["op"]
        if op not in ("fix_exposure", "denoise", "sharpen"):
            raise ValueError("op must be fix_exposure|denoise|sharpen")
        _STATE["img"], ev = tools.TOOLS[op](_STATE["img"])
        return ev
    if name == "run_full_inspection":
        r = service.inspect(_STATE["img"])
        r.pop("input_png"), r.pop("overlay_png")
        return r
    raise ValueError(f"unknown tool {name}")


def handle(req):
    m, rid = req.get("method"), req.get("id")
    if m == "initialize":
        return {"jsonrpc": "2.0", "id": rid, "result": {"protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}}, "serverInfo": {"name": "visionpilot", "version": "1.0"}}}
    if m == "tools/list":
        return {"jsonrpc": "2.0", "id": rid, "result": {"tools": TOOL_DEFS}}
    if m == "tools/call":
        p = req.get("params", {})
        try:
            out = call(p["name"], p.get("arguments", {}))
            return {"jsonrpc": "2.0", "id": rid, "result": {"content": [{"type": "text", "text": json.dumps(out)}]}}
        except Exception as e:  # noqa: BLE001
            return {"jsonrpc": "2.0", "id": rid, "result": {"isError": True, "content": [{"type": "text", "text": str(e)}]}}
    if rid is None:  # notification
        return None
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": f"method not found: {m}"}}


if __name__ == "__main__":
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        resp = handle(json.loads(line))
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
