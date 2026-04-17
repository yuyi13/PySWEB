"""ERA5-Land unit conversion and short-reference ET helpers for pysweb."""
from __future__ import annotations

import numpy as np

ALBEDO_SHORT = 0.23
CN_SHORT = 900.0
CD_SHORT = 0.34
EPS = np.finfo(float).tiny


def _as_float_array(value):
    return np.asarray(value, dtype=float)


def kelvin_to_celsius(value):
    return _as_float_array(value) - 273.15


def meters_to_mm_day(value):
    return _as_float_array(value) * 1000.0


def j_per_m2_to_mj_per_m2_day(value):
    return _as_float_array(value) / 1_000_000.0


def wind_speed_from_uv(u_component, v_component):
    u = _as_float_array(u_component)
    v = _as_float_array(v_component)
    return np.sqrt(np.square(u) + np.square(v))


def actual_vapor_pressure_from_dewpoint_c(dewpoint_c):
    dewpoint = _as_float_array(dewpoint_c)
    return 0.6108 * np.exp((17.27 * dewpoint) / (dewpoint + 237.3))


def _air_pressure_from_elevation_kpa(elev_m):
    elev = _as_float_array(elev_m)
    return 101.3 * np.power((293.0 - 0.0065 * elev) / 293.0, 9.8 / (0.0065 * 286.9))


def _psychrometric_constant_kpa_c(pair_kpa):
    return _as_float_array(pair_kpa) * 0.000665


def _saturation_vapor_pressure_c(temperature_c):
    temp = _as_float_array(temperature_c)
    return 0.6108 * np.exp((17.27 * temp) / (temp + 237.3))


def _slope_saturation_vapor_pressure_curve(temperature_c):
    temp = _as_float_array(temperature_c)
    es = _saturation_vapor_pressure_c(temp)
    return 4098.0 * es / np.square(temp + 237.3)


def _declination_rad(doy):
    doy_arr = _as_float_array(doy)
    return 0.409 * np.sin((2.0 * np.pi * doy_arr / 365.0) - 1.39)


def _inverse_earth_sun_distance(doy):
    doy_arr = _as_float_array(doy)
    return 1.0 + 0.033 * np.cos(2.0 * np.pi * doy_arr / 365.0)


def _sunset_hour_angle_rad(lat_rad, declination_rad):
    cos_ws = -np.tan(lat_rad) * np.tan(declination_rad)
    return np.arccos(np.clip(cos_ws, -1.0, 1.0))


def _extraterrestrial_radiation_mj_m2_day(lat_deg, doy):
    lat_rad = np.deg2rad(_as_float_array(lat_deg))
    decl = _declination_rad(doy)
    omega_s = _sunset_hour_angle_rad(lat_rad, decl)
    theta = (
        omega_s * np.sin(lat_rad) * np.sin(decl)
        + np.cos(lat_rad) * np.cos(decl) * np.sin(omega_s)
    )
    return (24.0 / np.pi) * 4.92 * _inverse_earth_sun_distance(doy) * theta


def _clear_sky_radiation_mj_m2_day(lat_deg, ea_kpa, pair_kpa, ra_mj_m2_day, doy):
    lat_rad = np.deg2rad(_as_float_array(lat_deg))
    ea = _as_float_array(ea_kpa)
    pair = _as_float_array(pair_kpa)
    ra = _as_float_array(ra_mj_m2_day)
    sin_beta_24 = np.sin(
        lat_rad * np.sin((2.0 * np.pi * _as_float_array(doy) / 365.0) - 1.39) * 0.3
        + 0.85
        - np.square(lat_rad) * 0.42
    )
    sin_beta_24 = np.maximum(sin_beta_24, 0.1)
    precipitable_water = pair * 0.14 * ea + 2.1
    kb = np.exp(
        -0.075 * np.power(precipitable_water / sin_beta_24, 0.4)
        - 0.00146 * pair / sin_beta_24
    ) * 0.98
    kd = np.minimum(kb * -0.36 + 0.35, kb * 0.82 + 0.18)
    return (kb + kd) * ra


def _net_longwave_radiation_mj_m2_day(tmax_c, tmin_c, ea_kpa, rs_mj_m2_day, rso_mj_m2_day):
    tmax = _as_float_array(tmax_c)
    tmin = _as_float_array(tmin_c)
    ea = np.maximum(_as_float_array(ea_kpa), 0.0)
    rs = _as_float_array(rs_mj_m2_day)
    rso = np.maximum(_as_float_array(rso_mj_m2_day), EPS)
    cloudiness = 1.35 * np.clip(rs / rso, 0.3, 1.0) - 0.35
    return (
        4.901e-9
        * cloudiness
        * (0.34 - 0.14 * np.sqrt(ea))
        * 0.5
        * (np.power(tmax + 273.16, 4) + np.power(tmin + 273.16, 4))
    )


def compute_daily_eto_short(
    tmax_c,
    tmin_c,
    ea_kpa,
    rs_mj_m2_day,
    uz_m_s,
    zw_m,
    elev_m,
    lat_deg,
    doy,
):
    tmax = _as_float_array(tmax_c)
    tmin = _as_float_array(tmin_c)
    ea = _as_float_array(ea_kpa)
    rs = _as_float_array(rs_mj_m2_day)
    uz = _as_float_array(uz_m_s)
    zw = _as_float_array(zw_m)
    elev = _as_float_array(elev_m)
    lat = _as_float_array(lat_deg)
    doy_arr = _as_float_array(doy)

    tmean = 0.5 * (tmax + tmin)
    pair_kpa = _air_pressure_from_elevation_kpa(elev)
    psy_kpa_c = _psychrometric_constant_kpa_c(pair_kpa)
    delta_kpa_c = _slope_saturation_vapor_pressure_curve(tmean)
    es_kpa = 0.5 * (
        _saturation_vapor_pressure_c(tmax) + _saturation_vapor_pressure_c(tmin)
    )
    vpd_kpa = np.maximum(es_kpa - ea, 0.0)
    u2_m_s = uz * 4.87 / np.log(67.8 * zw - 5.42)

    ra_mj_m2_day = _extraterrestrial_radiation_mj_m2_day(lat, doy_arr)
    rso_mj_m2_day = _clear_sky_radiation_mj_m2_day(lat, ea, pair_kpa, ra_mj_m2_day, doy_arr)
    rns_mj_m2_day = (1.0 - ALBEDO_SHORT) * rs
    rnl_mj_m2_day = _net_longwave_radiation_mj_m2_day(tmax, tmin, ea, rs, rso_mj_m2_day)
    rn_mj_m2_day = rns_mj_m2_day - rnl_mj_m2_day

    numerator = 0.408 * delta_kpa_c * rn_mj_m2_day + psy_kpa_c * (
        CN_SHORT / (tmean + 273.0)
    ) * u2_m_s * vpd_kpa
    denominator = delta_kpa_c + psy_kpa_c * (1.0 + CD_SHORT * u2_m_s)
    return numerator / denominator
