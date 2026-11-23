"""Rare-incident triage for Singapore traffic cameras.

Pipeline: collect -> labels -> embed -> features -> rank -> pool -> (vlm, review) -> evaluate -> site
Run `python -m sgtriage --help` for the steps.
"""
__version__ = "0.1.0"
