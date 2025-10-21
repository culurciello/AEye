from onvif import ONVIFCamera
import threading
import time
from datetime import datetime
import os

# Camera list
CAMERAS = [
    {"name": "frnt", "ip": "192.168.6.254", "user": "admin", "pass": "tokkigeo1"},
    {"name": "back", "ip": "192.168.6.255", "user": "admin", "pass": "tokkigeo1"},
    {"name": "entr", "ip": "192.168.7.1", "user": "admin", "pass": "tokkigeo1"}
]

PULL_INTERVAL = 5  # seconds
LOG_DIR = "camera_events"
os.makedirs(LOG_DIR, exist_ok=True)

def log_event(camera_name, event_text):
    """Append event to log file with timestamp."""
    filename = os.path.join(LOG_DIR, f"{camera_name}.log")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(filename, "a") as f:
        f.write(f"[{timestamp}] {event_text}\n")

def pull_events(cam):
    while True:
        try:
            camera = ONVIFCamera(cam["ip"], 8000, cam["user"], cam["pass"])
            events_service = camera.create_events_service()
            
            # Create PullPoint subscription
            subscription = events_service.CreatePullPointSubscription()
            print(f"[{cam['name']}] Subscription created")

            while True:
                try:
                    # Pull messages from the subscription
                    pull = events_service.create_type('PullMessages')
                    pull.Timeout = 'PT5S'
                    pull.MessageLimit = 10
                    pull.SubscriptionReference = subscription.SubscriptionReference
                    messages = events_service.PullMessages(pull)

                    for msg in getattr(messages, 'NotificationMessage', []):
                        log_event(cam['name'], str(msg))
                        print(f"[{cam['name']}] Event: {msg}")

                    time.sleep(PULL_INTERVAL)

                except Exception as e:
                    print(f"[{cam['name']}] Error pulling messages: {e}")
                    time.sleep(PULL_INTERVAL)

        except Exception as e:
            print(f"[{cam['name']}] Error setting up subscription: {e}")
            time.sleep(10)  # Retry after 10 seconds if setup fails

if __name__ == "__main__":
    threads = []
    for cam in CAMERAS:
        t = threading.Thread(target=pull_events, args=(cam,), daemon=True)
        t.start()
        threads.append(t)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping event pulling...")
