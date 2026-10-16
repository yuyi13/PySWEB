#!/usr/bin/env python3
"""
Script: test_multisensor_quality.py
Objective: Check sensor-specific band mappings, unused cirrus bits and Landsat 7 dropped-pixel handling.
Author: Yi Yu (with assistance from Codex)
Created: 2026-10-17
Last updated: 2026-10-17
Inputs: Independently specified Collection 2 bit-field fixtures.
Outputs: pytest assertions.
Usage: pytest tests/ssebop/test_multisensor_quality.py
Dependencies: numpy, pytest, pysweb.
"""
import pytest
from pysweb.ssebop.quality import landsat_c2_qa_mask, landsat_c2_l2_bands


def test_sensor_bands_and_unused_bit():
    for sensor in ('LT05','LE07'):
        assert landsat_c2_l2_bands(sensor)==('ST_B6','SR_B3','SR_B4')
        assert landsat_c2_qa_mask([64,68,72,192],[0,0,0,0],sensor=sensor).tolist()==[True,True,False,False]
    for sensor in ('LC08','LC09'):
        assert landsat_c2_l2_bands(sensor)==('ST_B10','SR_B4','SR_B5')
        assert landsat_c2_qa_mask([64,68],[0,0],sensor=sensor).tolist()==[True,False]
    assert not landsat_c2_qa_mask([64],[512],sensor='LE07').item()
    with pytest.raises(ValueError,match='Unsupported'):
        landsat_c2_qa_mask([64],[0],sensor='unknown')
