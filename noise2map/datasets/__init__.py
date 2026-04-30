from .whu_cd import WHUChangeDetectionDataset
from .whu_sem import WHUSegmentationDataset
from .xview2_cd import XView2WildfireCDDataset
from .xview2_sem import XView2WildfireSemDataset
from .spacenet7_cd import SpaceNet7CDDataset
from .spacenet7_sem import SN7MAPPING

__all__ = [
    "WHUChangeDetectionDataset",
    "WHUSegmentationDataset",
    "XView2WildfireCDDataset",
    "XView2WildfireSemDataset",
    "SpaceNet7CDDataset",
    "SN7MAPPING",
]
