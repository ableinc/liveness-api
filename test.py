"""
Ad-hoc test script for the liveness API.

Sends 3 image files, in the fixed order the API expects (front-facing eyes
open, head turned right, head turned left), to an already-running instance
(see README for how to start it) and prints the response for manual
inspection. Unlike moderation-api's test.py, there's no meaningful way to
hardcode "spoof" vs "live" scenarios as literals — feed it real captures,
e.g. yourself doing the actual front/right/left sequence in front of a
webcam (should come back live=true), and a printed photo or phone screen
physically rotated to fake each pose (should come back live=false).

Usage:
    python test.py front.jpg right.jpg left.jpg
"""

import base64
import json
import sys
import urllib.error
import urllib.request

BASE_URL = "http://127.0.0.1:9091"


def call_liveness_api(frame_paths: list[str]) -> dict:
    frames_b64 = []
    for path in frame_paths:
        with open(path, "rb") as f:
            frames_b64.append(base64.b64encode(f.read()).decode("ascii"))

    payload = json.dumps({"frames": frames_b64}).encode("utf-8")
    request = urllib.request.Request(
        f"{BASE_URL}/v1/liveness",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    if len(sys.argv) != 4:
        print(f"Usage: python {sys.argv[0]} front.jpg right.jpg left.jpg")
        sys.exit(1)

    frame_paths = sys.argv[1:4]
    print(f"Testing liveness API at {BASE_URL} with (front, right, left): {frame_paths}\n")

    try:
        result = call_liveness_api(frame_paths)
    except urllib.error.URLError as exc:
        print(f"ERROR - could not reach API ({exc})")
        sys.exit(1)
    except urllib.error.HTTPError as exc:
        print(f"HTTP {exc.code} - {exc.read().decode('utf-8')}")
        sys.exit(1)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
