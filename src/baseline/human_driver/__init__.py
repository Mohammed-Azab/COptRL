# Human driver baseline — rule-based speed planner for the quarter-car env.
#
# Looks ahead up to preview_m metres for upcoming bumps, figures out the
# fastest safe crossing speed from bump slope (π·H/W), then starts braking at
# exactly the right distance to arrive at that speed. Crosses the bump and
# re-accelerates. Pure geometry and kinematics — nothing fancy.
#
# Parameters live in config/baseline/human_driver_params.yaml.