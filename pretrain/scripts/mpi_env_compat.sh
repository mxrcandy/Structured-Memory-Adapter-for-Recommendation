#!/usr/bin/env bash

# Build environment forwarding arguments for the MPI implementation available
# on the current host. OpenMPI uses -x, while Intel MPI uses -genv.
mpi_env_args() {
  local variable value
  MPI_ENV_ARGS=()
  if mpirun --version 2>&1 | grep -qi 'Intel.*MPI'; then
    for variable in "$@"; do
      if [[ -v "$variable" ]]; then
        value="${!variable}"
      else
        value=""
      fi
      MPI_ENV_ARGS+=(-genv "$variable" "$value")
    done
  else
    for variable in "$@"; do
      MPI_ENV_ARGS+=(-x "$variable")
    done
  fi
}
