from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .state_machine import EmbodiedPickPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run embodied pick pipeline once.")
    parser.add_argument("--config", default="configs/default_config.json")
    parser.add_argument("--command", default="E", help="BCI command, e.g. A/B/C/E/H or 1-8")
    parser.add_argument("--frame-dir", default=None, help="Directory containing color.png/depth.png/workspace_mask.png/meta.mat")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    pipeline = EmbodiedPickPipeline(config)
    if args.command.upper() == "E" and pipeline.router.confirmed_target is None:
        pipeline.router.confirmed_target = pipeline.router.highlighted_target
    context = pipeline.handle_bci_command(args.command, frame_dir=args.frame_dir)
    payload = context.as_dict()
    print(json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None))


if __name__ == "__main__":
    main()