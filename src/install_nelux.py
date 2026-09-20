"""Install the latest Nelux release wheel matching Python, platform and torch ABI."""

import json
import os
import subprocess
import sys
import urllib.request

from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename
from packaging.version import Version


def selectWheel(assets, torchVersion, compatibleTags):
    version = Version(torchVersion)
    torchBuild = (int(f"{version.major}{version.minor}"), "torch")
    matches = []
    for asset in assets:
        name = asset["name"]
        if not name.endswith(".whl"):
            continue
        distribution, _, build, tags = parse_wheel_filename(name)
        if distribution == "nelux" and build == torchBuild and tags & compatibleTags:
            matches.append(asset["browser_download_url"])
    if len(matches) != 1:
        raise RuntimeError(f"Expected one Nelux wheel matching Python/platform and torch {torchVersion}; found {len(matches)}. Available: {[a['name'] for a in assets]}")
    return matches[0]


def main():
    import torch
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "python-decoders-benchmarks"}
    if os.environ.get("GH_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['GH_TOKEN']}"
    request = urllib.request.Request("https://api.github.com/repos/NevermindNilas/Nelux/releases/latest", headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        release = json.load(response)
    url = selectWheel(release["assets"], torch.__version__, set(sys_tags()))
    print(f"Installing {release['tag_name']} for torch {torch.__version__}: {url}", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", url], check=True)
    subprocess.run([sys.executable, "-c", "import torch; from nelux import VideoReader; print(VideoReader)"], check=True)


if __name__ == "__main__":
    main()
