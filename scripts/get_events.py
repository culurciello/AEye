from onvif import ONVIFCamera
import time
import threading
from datetime import datetime

# -----------------------
# CONFIG: your cameras
# -----------------------
CAMERAS = [
    {"name": "frnt", "host": "192.168.6.254", "port": 8000, "user": "admin", "pass": "tokkigeo1"},
    {"name": "back", "host": "192.168.6.255", "port": 8000, "user": "admin", "pass": "tokkigeo1"},
    {"name": "entr", "host": "192.168.7.1", "port": 8000, "user": "admin", "pass": "tokkigeo1"},
]

# Polling settings
PULL_TIMEOUT = 60  # seconds (will be converted to ISO 8601)
MESSAGE_LIMIT = 10  # max messages per pull

# -----------------------
# FUNCTION: listen to a single camera
# -----------------------
def listen_camera(cam_info):
    host = cam_info['host']
    port = cam_info['port']
    user = cam_info['user']
    passwd = cam_info['pass']
    name = cam_info['name']
    
    pullpoint = None
    subscription = None
    
    while True:  # Outer loop for reconnection
        try:
            # Create camera connection
            print(f"[{name}:{host}] Connecting to camera...")
            cam = ONVIFCamera(host, port, user, passwd)
            
            # Create events service
            events_service = cam.create_events_service()
            
            # Get event properties first (optional but helpful)
            try:
                properties = events_service.GetEventProperties()
                print(f"[{name}:{host}] Event properties retrieved")
            except:
                print(f"[{name}:{host}] Could not get event properties (not critical)")
            
            # Create pull point subscription
            print(f"[{name}:{host}] Creating subscription...")
            subscription = events_service.CreatePullPointSubscription()
            
            # Create pullpoint service
            pullpoint = cam.create_pullpoint_service()
            
            print(f"[{name}:{host}] Successfully connected and subscribed!")
            print(f"[{name}:{host}] Listening for events...")
            
            # Inner loop for pulling messages
            consecutive_errors = 0
            while consecutive_errors < 5:
                try:
                    # Pull messages
                    messages = pullpoint.PullMessages({
                        'Timeout': f'PT{PULL_TIMEOUT}S',  # ISO 8601 duration format
                        'MessageLimit': MESSAGE_LIMIT
                    })
                    
                    # Reset error counter on success
                    consecutive_errors = 0
                    
                    # Process messages
                    if hasattr(messages, 'NotificationMessage') and messages.NotificationMessage:
                        for msg in messages.NotificationMessage:
                            process_message(name, host, msg)
                    
                except Exception as e:
                    error_str = str(e)
                    
                    # Check if it's a timeout (which is normal)
                    if 'timeout' in error_str.lower() or 'timed out' in error_str.lower():
                        # Timeout is normal when no events occur
                        continue
                    
                    # For other errors, increment counter
                    consecutive_errors += 1
                    print(f"[{name}:{host}] Pull error ({consecutive_errors}/5): {e}")
                    
                    if consecutive_errors >= 5:
                        print(f"[{name}:{host}] Too many errors, reconnecting...")
                        break
                    
                    time.sleep(2)
            
        except Exception as e:
            print(f"[{name}:{host}] Connection error: {e}")
            print(f"[{name}:{host}] Will retry in 10 seconds...")
            time.sleep(10)
        
        finally:
            # Clean up
            if pullpoint:
                try:
                    pullpoint.Unsubscribe()
                except:
                    pass

def process_message(name, host, msg):
    """Process a single notification message"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    # Extract topic
    topic = "Unknown"
    if hasattr(msg, 'Topic'):
        if hasattr(msg.Topic, '_value_1'):
            topic = msg.Topic._value_1
        else:
            topic = str(msg.Topic)
    
    # Extract message details
    details = {}
    if hasattr(msg, 'Message'):
        if hasattr(msg.Message, 'Message'):
            # Nested message structure
            inner_msg = msg.Message.Message
            if hasattr(inner_msg, 'Data'):
                if hasattr(inner_msg.Data, 'SimpleItem'):
                    for item in inner_msg.Data.SimpleItem:
                        if hasattr(item, 'Name') and hasattr(item, 'Value'):
                            details[item.Name] = item.Value
        elif hasattr(msg.Message, '_value_1'):
            details['raw'] = str(msg.Message._value_1)
    
    # Print formatted event
    print(f"\n[{timestamp}] [{name}:{host}] EVENT DETECTED!")
    print(f"  Topic: {topic}")
    if details:
        for key, value in details.items():
            print(f"  {key}: {value}")
    else:
        print(f"  Data: {msg}")
    print("")

# -----------------------
# ALTERNATIVE: Simple polling approach
# -----------------------
def listen_camera_simple(cam_info):
    """
    Simpler approach using basic event retrieval
    """
    host = cam_info['host']
    port = cam_info['port']
    user = cam_info['user']
    passwd = cam_info['pass']
    name = cam_info['name']
    
    while True:
        try:
            # Connect to camera
            cam = ONVIFCamera(host, port, user, passwd)
            
            # Try to get events directly
            events = cam.create_events_service()
            
            print(f"[{name}:{host}] Connected (simple mode)")
            
            # Try to get current event states
            while True:
                try:
                    # Try GetEventProperties to see current states
                    props = events.GetEventProperties()
                    print(f"[{name}:{host}] Properties check: {type(props)}")
                    
                    # Some cameras support GetServiceCapabilities
                    caps = events.GetServiceCapabilities()
                    print(f"[{name}:{host}] Capabilities: WSPullPointSupport={caps.WSPullPointSupport if hasattr(caps, 'WSPullPointSupport') else 'Unknown'}")
                    
                except Exception as e:
                    print(f"[{name}:{host}] Simple poll error: {e}")
                
                time.sleep(10)  # Check every 10 seconds
                
        except Exception as e:
            print(f"[{name}:{host}] Simple mode connection error: {e}")
            time.sleep(10)

# -----------------------
# DIAGNOSTIC: Test single camera
# -----------------------
def test_single_camera(host, port, user, passwd):
    """Test function to diagnose issues with a single camera"""
    print(f"\n=== Testing camera at {host}:{port} ===")
    
    try:
        # Connect
        print("1. Creating camera object...")
        cam = ONVIFCamera(host, port, user, passwd)
        print("   ✓ Camera object created")
        
        # Get device info
        print("2. Getting device information...")
        device_info = cam.devicemgmt.GetDeviceInformation()
        print(f"   ✓ Device: {device_info.Manufacturer} {device_info.Model}")
        
        # Create events service
        print("3. Creating events service...")
        events = cam.create_events_service()
        print("   ✓ Events service created")
        
        # Get capabilities
        print("4. Getting event capabilities...")
        try:
            caps = events.GetServiceCapabilities()
            print(f"   ✓ Capabilities retrieved")
            print(f"     - WSPullPointSupport: {caps.WSPullPointSupport if hasattr(caps, 'WSPullPointSupport') else 'N/A'}")
            print(f"     - WSSubscriptionPolicySupport: {caps.WSSubscriptionPolicySupport if hasattr(caps, 'WSSubscriptionPolicySupport') else 'N/A'}")
        except Exception as e:
            print(f"   ⚠ Could not get capabilities: {e}")
        
        # Create subscription
        print("5. Creating pull point subscription...")
        subscription = events.CreatePullPointSubscription()
        print(f"   ✓ Subscription created: {type(subscription)}")
        
        # Check subscription attributes
        print("6. Checking subscription attributes...")
        attrs = dir(subscription)
        print(f"   ✓ Subscription has {len(attrs)} attributes")
        
        # Look for useful attributes
        useful_attrs = [a for a in attrs if 'pull' in a.lower() or 'message' in a.lower()]
        if useful_attrs:
            print(f"     Relevant attributes: {useful_attrs}")
        
        # Try to pull messages directly from subscription
        print("7. Attempting to pull messages...")
        try:
            if hasattr(subscription, 'PullMessages'):
                messages = subscription.PullMessages({'Timeout': 'PT5S', 'MessageLimit': 10})
                print(f"   ✓ PullMessages worked directly!")
            else:
                print(f"   ⚠ Subscription has no PullMessages method")
                
                # Try pullpoint service
                print("8. Creating pullpoint service...")
                pullpoint = cam.create_pullpoint_service()
                print(f"   ✓ Pullpoint service created: {type(pullpoint)}")
                
                print("9. Pulling from pullpoint service...")
                messages = pullpoint.PullMessages({'Timeout': 'PT5S', 'MessageLimit': 10})
                print(f"   ✓ PullMessages worked via pullpoint service!")
                
        except Exception as e:
            print(f"   ✗ Pull failed: {e}")
        
        print("\n=== Test complete ===\n")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}\n")
        return False

# -----------------------
# MAIN: launch a thread for each camera
# -----------------------
def main():
    print("ONVIF Event Listener")
    print("=" * 50)
    
    # Optional: Test first camera before starting threads
    if False:  # Change to True to run diagnostic
        first_cam = CAMERAS[0]
        test_single_camera(first_cam['host'], first_cam['port'], 
                          first_cam['user'], first_cam['pass'])
        input("Press Enter to continue with normal operation...")
    
    threads = []
    
    print(f"Starting event listeners for {len(CAMERAS)} cameras...")
    print("Press Ctrl+C to exit\n")
    
    for cam in CAMERAS:
        # Use main approach
        t = threading.Thread(target=listen_camera, args=(cam,), daemon=True)
        t.start()
        threads.append(t)
    
    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down event listeners...")
        print("Exiting...")

if __name__ == "__main__":
    main()