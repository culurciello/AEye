#!/usr/bin/env python3
from onvif import ONVIFCamera

# List your cameras here
CAMERAS = [
    {"name": "frnt", "ip": "192.168.6.254", "port": 8000, "user": "admin", "pass": "tokkigeo1"},
    {"name": "back", "ip": "192.168.6.255", "port": 8000, "user": "admin", "pass": "tokkigeo1"},
    {"name": "entr", "ip": "192.168.7.1", "port": 8000, "user": "admin", "pass": "tokkigeo1"},
]

def check_camera(cam):
    print(f"\n=== Checking {cam['name']} ({cam['ip']}) ===")

    try:
        # Connect to camera
        camera = ONVIFCamera(cam['ip'], cam['port'], cam['user'], cam['pass'])

        # Get device capabilities
        dev_service = camera.create_devicemgmt_service()
        capabilities = dev_service.GetCapabilities()
        print("Capabilities:")
        print(capabilities)

        # Media service: check profiles and snapshot URIs
        media_service = camera.create_media_service()
        profiles = media_service.GetProfiles()
        print(f"Media profiles ({len(profiles)}):")
        for p in profiles:
            print(f"  - Name: {p.Name}, Token: {p.token}")
            try:
                uri = media_service.GetSnapshotUri({'ProfileToken': p.token}).Uri
                print(f"    Snapshot URI: {uri}")
            except:
                print("    Snapshot URI not supported")

        # Check for PTZ support
        try:
            ptz_service = camera.create_ptz_service()
            ptz_nodes = ptz_service.GetNodes()
            if ptz_nodes:
                print(f"PTZ supported: {len(ptz_nodes)} nodes")
            else:
                print("PTZ not supported")
        except:
            print("PTZ service not available")

        # Check events support
        try:
            events_service = camera.create_events_service()
            print("Events service available")
        except:
            print("Events service not available")

    except Exception as e:
        print(f"Error connecting to camera {cam['name']}: {e}")

def main():
    for cam in CAMERAS:
        check_camera(cam)

if __name__ == "__main__":
    main()
