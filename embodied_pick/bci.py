from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class BCICommand:
    raw: str
    name: str
    description: str


COMMANDS: Dict[str, BCICommand] = {
    "A": BCICommand("A", "previous_target", "switch highlight to previous target"),
    "B": BCICommand("B", "next_target", "switch highlight to next target"),
    "C": BCICommand("C", "confirm_target", "confirm highlighted target"),
    "D": BCICommand("D", "cancel", "cancel current selection"),
    "E": BCICommand("E", "execute", "start execution"),
    "F": BCICommand("F", "continue_clarify", "continue clarification"),
    "G": BCICommand("G", "execute_with_current_info", "stop clarification and execute"),
    "H": BCICommand("H", "emergency_stop", "pause or emergency stop"),
}


class BCICommandRouter:
    def __init__(self, targets: Optional[List[str]] = None) -> None:
        self.targets = targets or ["cup", "medicine_box", "tissue_box", "phone"]
        self.index = 0
        self.confirmed_target: Optional[str] = None

    @property
    def highlighted_target(self) -> str:
        return self.targets[self.index]

    def route(self, raw: object) -> BCICommand:
        key = str(raw).strip().upper()
        if key.isdigit():
            key = self._index_to_key(int(key))
        return COMMANDS.get(key, BCICommand(key, "unknown", "unknown platform command"))

    def apply(self, raw: object) -> BCICommand:
        command = self.route(raw)
        if command.name == "previous_target":
            self.index = (self.index - 1) % len(self.targets)
        elif command.name == "next_target":
            self.index = (self.index + 1) % len(self.targets)
        elif command.name == "confirm_target":
            self.confirmed_target = self.highlighted_target
        elif command.name == "cancel":
            self.confirmed_target = None
        return command

    @staticmethod
    def _index_to_key(index: int) -> str:
        keys = list(COMMANDS.keys())
        if 1 <= index <= len(keys):
            return keys[index - 1]
        return str(index)