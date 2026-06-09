# Road profile generator.
# Profiles: speed_bump, flat, recorded.
#
#   RoadGenerator(profile, vehicle_speed)   — fixed or recorded road
#   RoadGenerator.from_random(rng, speed)   — random bump layout each episode
#   RoadGenerator.from_scenario_file(path)  — load a named JSON scenario
