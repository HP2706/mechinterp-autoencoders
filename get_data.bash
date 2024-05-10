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
    ;;
  *)
    echo "Unsupported OS: $OS" >&2
    exit 1
    ;;
esac


# Check if model name is provided as an argument
if [ -z "$1" ]; then
  echo "Error: No model name provided." >&2
  exit 1
fi

# Execute the command with the provided model name
modal volume get autoencoder laion2b_autoencoders/$1 --force
echo "Operation completed successfully."


