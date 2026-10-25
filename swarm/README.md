# Swarm simulation — ARGoS3 + Buzz

A 20-robot foot-bot flocking demo. Each robot runs the same Buzz script and uses
range-and-bearing sensing to compute a Lennard-Jones interaction with its
neighbors: attractive when far, repulsive when close. The swarm self-organizes
into an evenly spaced lattice.

## Prerequisites

ARGoS3 + Buzz installed (the `argos` module of `setup/setup_robotics.sh`):

```bash
./setup/setup_robotics.sh argos
```

Verify:
```bash
argos3 --version      # ARGOS_VERSION=3.0.0-beta... , ARGOS_WITH_LUA=ON
bzzc -h               # Buzz compiler
```

## Compile & run

```bash
# Compile the Buzz script to bytecode (the -I include path is required).
bzzc -I /usr/local/share/buzz/include -b flocking.bo -d flocking.bdbg flocking.bzz

# Launch — opens the QtOpenGL window; press ▶ to start the ticks.
argos3 -c flocking.argos
```

You should see 20 green foot-bots drift from a random cluster into a spaced lattice.

## Tuning

- `TARGET` (in `flocking.bzz`) — desired inter-robot spacing (cm).
- `EPSILON` — interaction strength.
- `quantity` (in `flocking.argos`) — swarm size.
- `rab_range` — how far each robot can sense neighbors.

## Notes

- `bzzc` uses flag-based args (`-b out.bo -d out.bdbg in.bzz`), not positional.
- Without `-I /usr/local/share/buzz/include`, `bzzc` can't find `vec2.bzz`.
- Compiled `.bo` / `.bdbg` are build artifacts (gitignored); regenerate with the command above.
