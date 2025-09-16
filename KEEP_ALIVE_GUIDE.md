# AEye Keep-Alive System Guide

This guide explains how to keep your AEye motion detection system running reliably, even when cameras become unavailable.

## **Two-Layer Protection:**

### **1. Application-Level Resilience** ✅
The main.py now includes:
- **Camera reconnection logic**: Automatically retries connection up to 10 times on startup
- **Runtime recovery**: Detects camera failures during operation and attempts reconnection
- **Graceful degradation**: Continues attempting to reconnect rather than crashing

### **2. Process-Level Keep-Alive** ✅
Two keep-alive scripts monitor and restart the process:

## **Quick Start:**

### **Basic Keep-Alive (Recommended for most users):**
```bash
# Simple keep-alive with automatic restart
python3 keep_alive.py
```

### **Advanced Keep-Alive (For production servers):**
```bash
# Advanced monitoring with health checks and notifications
python3 keep_alive_advanced.py
```

### **Background Processing:**
```bash
# Run in background with logs
nohup python3 keep_alive.py > keep_alive.log 2>&1 &

# Or with advanced version
nohup python3 keep_alive_advanced.py > keep_alive_advanced.log 2>&1 &
```

## **Script Features:**

### **keep_alive.py** (Simple)
- ✅ Automatic process restart on crash
- ✅ Rate limiting (max 5 restarts in 5 minutes)
- ✅ Graceful shutdown handling
- ✅ Real-time log forwarding
- ✅ Signal handling (Ctrl+C, SIGTERM)

### **keep_alive_advanced.py** (Production)
- ✅ All basic features plus:
- ✅ Configurable restart strategies for different failure types
- ✅ Exponential backoff on repeated failures
- ✅ Health monitoring based on log patterns
- ✅ Email notifications on failures
- ✅ Rotating log files
- ✅ Different delays for camera vs general failures

## **Linux Server Installation:**

### **Option 1: Systemd Service (Recommended for servers)**
```bash
# Install as system service
sudo ./install_service.sh

# Service management
sudo systemctl status aeye    # Check status
sudo systemctl stop aeye     # Stop
sudo systemctl start aeye    # Start
sudo systemctl restart aeye  # Restart
sudo journalctl -u aeye -f   # View live logs
```

### **Option 2: Manual Background Process**
```bash
# Create screen session
screen -S aeye
python3 keep_alive.py
# Press Ctrl+A, then D to detach

# Reattach later
screen -r aeye
```

## **Configuration:**

### **Basic Configuration (keep_alive.py)**
Edit the script to modify:
```python
COMMAND = ["python3", "main.py", "--video-source", "rtsp://your-camera", "--headless"]
RESTART_DELAY = 5  # seconds
MAX_RESTART_ATTEMPTS = 5
RESTART_WINDOW = 300  # 5 minutes
```

### **Advanced Configuration (keep_alive_advanced.py)**
Creates `keep_alive_config.json`:
```json
{
  "command": ["python3", "main.py", "--video-source", "rtsp://192.168.6.244:554/11", "--headless"],
  "restart_delay": 5,
  "failure_strategies": {
    "camera_failure": {
      "delay": 30,
      "max_attempts": 10,
      "backoff_multiplier": 1.5
    }
  },
  "notifications": {
    "enabled": false,
    "email": {
      "smtp_server": "smtp.gmail.com",
      "username": "your-email@gmail.com",
      "to_address": "admin@yourdomain.com"
    }
  }
}
```

## **Camera Failure Scenarios Handled:**

### **1. Startup Failures**
- Network camera not responding
- Invalid RTSP URL
- Authentication issues
- → **Result**: Retries up to 10 times with 5s delays

### **2. Runtime Disconnections**
- Camera power loss
- Network interruption
- RTSP stream timeout
- → **Result**: Attempts reconnection 5 times, then exits for keep-alive restart

### **3. Intermittent Issues**
- Occasional frame drops
- Temporary network glitches
- → **Result**: Tolerates up to 30 consecutive failed reads before reconnecting

## **Monitoring:**

### **Log Files:**
- `data/keep_alive.log` - Keep-alive script logs
- `data/aeye.log` - Main application logs

### **Health Indicators:**
Watch for these log messages:
- ✅ `"Camera connection successful"`
- ✅ `"Motion-triggered processor started"`
- ⚠️ `"Failed to read frame"`
- ⚠️ `"Camera reconnection failed"`
- 🔄 `"Process restarted"`

### **System Resources:**
```bash
# Check process status
ps aux | grep python3

# Monitor resource usage
top -p $(pgrep -f "python3.*main.py")

# Check disk space (for videos)
df -h data/
```

## **Troubleshooting:**

### **Camera Won't Connect:**
```bash
# Test camera directly
ffmpeg -i rtsp://192.168.6.244:554/11 -t 10 test.mp4

# Test with VLC
vlc rtsp://192.168.6.244:554/11
```

### **High Restart Rate:**
1. Check camera stability
2. Verify network connection
3. Increase restart delays in config
4. Check system resources (CPU, memory, disk)

### **Process Not Restarting:**
1. Check keep-alive script logs
2. Verify script permissions (`chmod +x keep_alive.py`)
3. Check if max restart limit was reached

## **Best Practices:**

1. **Use headless mode** for servers: `--headless`
2. **Monitor disk space** for video recordings
3. **Set up log rotation** for long-running deployments
4. **Test camera reliability** before production
5. **Use systemd service** for automatic startup on boot
6. **Configure email notifications** for critical systems

This keep-alive system ensures your AEye motion detection runs continuously, automatically recovering from camera failures and system issues!