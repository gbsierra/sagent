"""Where a catalog fact came from, and when someone last checked it.

A catalog value is only as good as the page it was copied from. Each
:class:`~sagent.catalog.table.ModelTable` names a :class:`Source` per
:data:`Fact` group, and may override it for single rows, so a reviewer
seeing a disagreement knows which page to reopen and how old the reading is.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Final, Literal, cast, get_args


__all__ = ["FACTS", "Fact", "Source"]


type Fact = Literal["limits", "prices", "thinking"]
"""A group of row fields that one vendor page states together."""

FACTS: Final[tuple[Fact, ...]] = cast(
    tuple[Fact, ...],
    get_args(cast(object, Fact.__value__)),
)


@dataclass(frozen=True, slots=True)
class Source:
    """The page a fact group was read from."""

    url: str

    checked: date | None = None
    """When the values were last compared against ``url``; ``None`` when that
    date was not recorded."""
