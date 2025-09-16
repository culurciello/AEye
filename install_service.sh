#!/bin/bash

# Install script for AEye systemd service
# Run this on Linux servers to install AEye as a system service

set -e

# Configuration
SERVICE_NAME="aeye"
SERVICE_USER="aeye"
SERVICE_GROUP="aeye"
INSTALL_DIR="/opt/aeye"
SERVICE_FILE="aeye.service"

echo "Installing AEye Motion Detection Service..."

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root (use sudo)"
   exit 1
fi

# Create service user
if ! id "$SERVICE_USER" &>/dev/null; then
    echo "Creating service user: $SERVICE_USER"
    useradd --system --shell /bin/false --home-dir /nonexistent --no-create-home $SERVICE_USER
fi

# Create install directory
echo "Creating installation directory: $INSTALL_DIR"
mkdir -p $INSTALL_DIR
mkdir -p $INSTALL_DIR/data

# Copy files
echo "Copying AEye files..."
cp -r * $INSTALL_DIR/
chown -R $SERVICE_USER:$SERVICE_GROUP $INSTALL_DIR

# Install Python dependencies
echo "Installing Python dependencies..."
pip3 install -r $INSTALL_DIR/requirements.txt

# Install systemd service
echo "Installing systemd service..."
cp $SERVICE_FILE /etc/systemd/system/
systemctl daemon-reload

# Enable and start service
echo "Enabling and starting service..."
systemctl enable $SERVICE_NAME
systemctl start $SERVICE_NAME

# Show status
echo ""
echo "Installation complete!"
echo ""
echo "Service status:"
systemctl status $SERVICE_NAME --no-pager

echo ""
echo "Useful commands:"
echo "  sudo systemctl status $SERVICE_NAME    # Check status"
echo "  sudo systemctl stop $SERVICE_NAME     # Stop service"
echo "  sudo systemctl start $SERVICE_NAME    # Start service"
echo "  sudo systemctl restart $SERVICE_NAME  # Restart service"
echo "  sudo journalctl -u $SERVICE_NAME -f   # View logs"
echo ""
echo "Configuration files:"
echo "  $INSTALL_DIR/keep_alive_config.json   # Keep-alive settings"
echo "  $INSTALL_DIR/data/                     # Data directory"