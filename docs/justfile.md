# Justfile recipes and the `dado` CLI

The workspace `Justfile` wraps the most common dev flows so onboarding
doesn't have to remember feature flags or backend selection. Install
[`just`](https://just.systems) (`brew install just`, `cargo install
just`) and run `just` for the list. Cross-OS where it makes sense:

```sh
just deps              # Install ngspice via brew/apt/dnf/pacman/zypper.
just deps-docker       # Build every rlx-eda image (ngspice, yosys, magic, klayout, ORFS).
just deps-docker-ngspice  # Build a single image: ngspice.
just deps-docker-yosys    # …or yosys, magic, klayout, orfs.
just deps-docker-check    # Smoke-test every image (--version).
just build             # cargo build --workspace
just test              # cargo test --workspace
just lint              # cargo clippy --all-targets -- -D warnings
just dado r2r          # = ./scripts/dado r2r
just dado sar --docker # = ./scripts/dado sar --docker
just run-dado-r2r      # raw cargo equivalent (no progress bars on pipes)
just run-dado-sar      # raw cargo equivalent
just run-dado-sar-docker  # raw cargo equivalent w/ Docker backend
```

## `dado` CLI wrapper — recommended entry point

For the DADO experiments specifically, `scripts/dado` is a single-entry
wrapper. It picks the right `cargo run` invocation, sets the
`NGSPICE_BACKEND` env var if you ask for `--docker`, auto-builds the
ngspice Docker image on first use, and shows live progress bars (via
[`indicatif`](https://docs.rs/indicatif)) so a 25-minute SPICE run
isn't an opaque wait.

```sh
./scripts/dado r2r              # spike-dado-r2r            (~30 s)
./scripts/dado sar              # spike-dado-sar            (~25 min, host ngspice)
./scripts/dado sar --docker     # same, with Docker ngspice (auto-builds image)
./scripts/dado all --docker     # both, in sequence
./scripts/dado help             # usage
```

`just dado <args>` is the same thing through the Justfile.

### Live output example

When invoked from a terminal, each phase shows a progress bar with
elapsed time, a counter, and an ETA. (Bars auto-suppress when stdout
is piped or redirected.)

```text
DADO vs EDA artifact run — output: crates/spike-dado-r2r/docs
config: K=100 n_iters=80 seeds=12 snapshots@[0, 9, 24, 49, 79]

[A] Synthetic target-matching ...
  [         synth] 00:00:01 ============>            240000/  800000 (00:00:03)
  [         synth] 00:00:04 =============================>  800000/  800000 (00:00:00)
  [synth] DADO 0.00000  vs  EDA -3.75000   (paired t = 17.23, p ≈ 0.0000)

[B] R-2R DAC max-INL ...
  [      R-2R INL] 00:00:18 =============>          312000/  800000 (00:00:28)
  …
```

For `spike-dado-sar`, the SPICE phase shows `[SPICE (local)]` or
`[SPICE (docker)]` depending on backend, ticking once per ngspice
invocation (~0.7 s on this machine). The ETA stays accurate because
ngspice eval cost is roughly constant across designs.
