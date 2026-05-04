"""
State-machine logic that converts band triggers into final detector states.

The rules here add persistence, release behaviour, and a safety override so the
output is less noisy and more meaningful for real monitoring.
"""

import pandas as pd

from .candidate_logic import rules_for_band


def dominant_state_rank(state: str) -> int:
    """
    Rank states numerically so the strongest state can be selected reliably.
    """
    return {"STABLE": 0, "WATCH": 1, "WARNING": 2, "ALARM": 3, "SAFETY": 4}.get(str(state), 0)


def _step_state(
    current: str,
    watch: bool,
    warning: bool,
    alarm: bool,
    counters: dict,
    rules,
):
    """
    Advance one band's state machine by a single frame using persistence counters.
    """
    counters = counters.copy()
    counters["watch"] = counters["watch"] + 1 if watch else 0
    counters["warning"] = counters["warning"] + 1 if warning else 0
    counters["alarm"] = counters["alarm"] + 1 if alarm else 0

    watch_ready = counters["watch"] >= rules.watch_persistence_frames
    warning_ready = counters["warning"] >= rules.warning_persistence_frames
    alarm_ready = counters["alarm"] >= rules.alarm_persistence_frames
    quiet = not watch

    if current == "STABLE":
        counters["release"] = 0
        if warning_ready:
            return "WARNING", counters
        if watch_ready:
            return "WATCH", counters
        return "STABLE", counters

    if current == "WATCH":
        if warning_ready:
            counters["release"] = 0
            return "WARNING", counters
        counters["release"] = counters["release"] + 1 if quiet else 0
        if counters["release"] >= rules.release_frames:
            counters["release"] = 0
            return "STABLE", counters
        return "WATCH", counters

    if current == "WARNING":
        if alarm_ready:
            counters["release"] = 0
            return "ALARM", counters
        counters["release"] = counters["release"] + 1 if quiet else 0
        if counters["release"] >= rules.release_frames:
            counters["release"] = 0
            return ("WATCH" if watch_ready else "STABLE"), counters
        return "WARNING", counters

    if current == "ALARM":
        counters["release"] = 0 if alarm else counters["release"] + 1
        if counters["release"] >= rules.release_frames:
            counters["release"] = 0
            if warning_ready:
                return "WARNING", counters
            if watch_ready:
                return "WATCH", counters
            return "STABLE", counters
        return "ALARM", counters

    return current, counters


def apply_state_machine(df: pd.DataFrame, safety_rms: float) -> pd.DataFrame:
    """
    Combine band triggers, persistence logic, and safety override into final states.
    """
    out = df.copy()

    states = {"third": "STABLE", "fifth": "STABLE"}
    counters = {
        "third": {"watch": 0, "warning": 0, "alarm": 0, "release": 0},
        "fifth": {"watch": 0, "warning": 0, "alarm": 0, "release": 0},
    }

    overall_states = []
    third_states = []
    fifth_states = []
    messages = []
    control = []

    for _, row in out.iterrows():
        safety = float(row["vib_rms_mean"]) >= float(safety_rms)

        for band in ("third", "fifth"):
            rules = rules_for_band(band)
            states[band], counters[band] = _step_state(
                current=states[band],
                watch=bool(row[f"{band}_watch_trigger"]),
                warning=bool(row[f"{band}_warning_trigger"]),
                alarm=bool(row[f"{band}_alarm_trigger"]),
                counters=counters[band],
                rules=rules,
            )

        if safety:
            state = "SAFETY"
        else:
            state = max((states["third"], states["fifth"]), key=dominant_state_rank)

        if state == "STABLE":
            msg, ctrl = "Stable condition.", "HOLD"
        elif state == "WATCH":
            msg, ctrl = "Precursor detected. Monitor closely.", "WATCH"
        elif state == "WARNING":
            msg, ctrl = "Confirmed instability warning.", "PREPARE"
        elif state == "ALARM":
            msg, ctrl = "Sustained chatter alarm.", "SLOWDOWN"
        else:
            msg, ctrl = "Safety threshold exceeded.", "SAFETY"

        third_states.append(states["third"])
        fifth_states.append(states["fifth"])
        overall_states.append(state)
        messages.append(msg)
        control.append(ctrl)

    out["third_state"] = third_states
    out["fifth_state"] = fifth_states
    out["state"] = overall_states
    out["state_message"] = messages
    out["control_action"] = control
    out["watch"] = (out["state"] == "WATCH").astype(int)
    out["warning"] = (out["state"] == "WARNING").astype(int)
    out["alarm"] = out["state"].isin(["ALARM", "SAFETY"]).astype(int)
    out["level"] = out["state"]

    return out