"""Domain model: geometry, annotations and generated samples.

These types are the contract between the renderers that produce pixels and the exporters
that write datasets. They are immutable and pixel-free, so they can be built, reordered
and serialised without touching an image.
"""

from .annotations import (
    Line,
    Page,
    Region,
    RegionType,
    Table,
    TableCell,
    Word,
    assign_reading_order,
    line_from_dict,
    page_from_dict,
    region_from_dict,
    table_from_dict,
    word_from_dict,
)
from .geometry import BBox, Point, Polygon
from .sample import Provenance, Sample

__all__ = [
    "BBox",
    "Line",
    "Page",
    "Point",
    "Polygon",
    "Provenance",
    "Region",
    "RegionType",
    "Sample",
    "Table",
    "TableCell",
    "Word",
    "assign_reading_order",
    "line_from_dict",
    "page_from_dict",
    "region_from_dict",
    "table_from_dict",
    "word_from_dict",
]
