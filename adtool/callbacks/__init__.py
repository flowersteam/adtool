from adtool.callbacks.base_callback import BaseCallback
from adtool.callbacks.custom_print_callback import CustomPrintCallback

"""Callback event payloads.

``on_discovery`` receives config, resource URI, run index, experiment ID,
seed, discovery, and optional rendered outputs. ``on_save`` receives the
saved pipeline and resource URI. ``on_save_finished`` receives the checkpoint
path as ``uid`` and report directory. ``on_finished`` and ``on_error`` receive
the run index, experiment ID, and seed.
"""
