#!/bin/bash

# Function to check if a command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Ensure necessary tools are installed
if ! command_exists modal; then
  echo "Error: 'modal' command is not installed." >&2
  exit 1
fi

# Detect the operating system
OS="$(uname -s)"
case "$OS" in
  Linux* | Darwin*)
    echo "Running on $OS"
    ;;
  *)
    echo "Unsupported OS: $OS" >&2
    exit 1
    ;;
esac

# Execute the command
modal volume get autoencoder clip_mechinterp_pipeline --force
echo "Operation completed successfully."

