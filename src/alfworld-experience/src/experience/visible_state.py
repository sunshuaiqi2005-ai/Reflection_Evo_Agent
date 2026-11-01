"""Working memory reconstructed only from visible action feedback within one episode."""
from __future__ import annotations

import re


def visible_state(steps):
    location, held = None, "unknown (use inventory if needed)"
    visited, searched, empty, processed = [], [], [], []
    latest_contents = None
    last_action, failure_streak = None, 0
    for step in steps:
        action, observation = step["action"], step["observation"].strip()
        if observation == "Nothing happens.":
            failure_streak = failure_streak + 1 if action == last_action else 1
            last_action = action
            continue
        failure_streak, last_action = 0, None
        arrival = re.search(r"You arrive at (.+?)\.", observation)
        if arrival:
            location = arrival.group(1)
            latest_contents = None
            if location not in visited:
                visited.append(location)
        pickup = re.search(r"You pick up the (.+?) from the (.+?)\.", observation)
        if pickup:
            held = pickup.group(1)
            latest_contents = None
        placement = re.search(r"You move the (.+?) to the (.+?)\.", observation)
        if placement:
            held = "empty"
            latest_contents = None
        if observation.startswith("You are carrying:"):
            held = observation.removeprefix("You are carrying:").strip().rstrip(".")
        elif observation == "You are not carrying anything.":
            held = "empty"
        if location and "you see" in observation and ("In it," in observation or "On the " in observation):
            latest_contents = observation
            if location not in searched:
                searched.append(location)
            if "you see nothing" in observation and location not in empty:
                empty.append(location)
            elif "you see nothing" not in observation and location in empty:
                empty.remove(location)
        for verb in ("heat", "clean", "cool"):
            match = re.search(r"You " + verb + r" the (.+?) using the .+?\.", observation)
            if match:
                item = verb + ": " + match.group(1)
                if item not in processed:
                    processed.append(item)
    return {"current_location": location or "middle of room; no successful navigation yet",
            "held_object": held, "visited_locations": visited, "searched_locations": searched,
            "observed_empty_locations": empty, "successful_processing": processed,
            "latest_contents_observation": latest_contents,
            "last_ineffective_action": last_action, "consecutive_ineffective_repeats": failure_streak}


def render_visible_state(state, token_count, budget):
    lines = [f"Current location: {state['current_location']}", f"Held object: {state['held_object']}"]
    if state["last_ineffective_action"]:
        lines.append(f"Last ineffective action: {state['last_ineffective_action']} ({state['consecutive_ineffective_repeats']} consecutive failures). Check preconditions and choose a different action.")
    extras = []
    if state["successful_processing"]:
        extras.append("Successful processing: " + "; ".join(state["successful_processing"]))
    if state["latest_contents_observation"]:
        extras.append("Last visible contents here: " + state["latest_contents_observation"])
    if state["observed_empty_locations"]:
        extras.append("Previously observed empty: " + ", ".join(state["observed_empty_locations"]))
    if state["searched_locations"]:
        extras.append("Previously inspected contents: " + ", ".join(state["searched_locations"]))
    if state["visited_locations"]:
        extras.append("Visited: " + ", ".join(state["visited_locations"]))
    for line in extras:
        if token_count("\n".join([*lines, line])) <= budget:
            lines.append(line)
    return "\n".join(lines)
