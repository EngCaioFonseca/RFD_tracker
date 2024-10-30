# test_ip_camera.py
import cv2
import time

def test_ip_camera(ip_address="10.5.0.2:8080"):
    """Test IP camera connection"""
    print(f"Testing IP camera connection at {ip_address}")
    
    # Construct camera URL
    camera_url = f"http://{ip_address}/video"
    print(f"Attempting to connect to: {camera_url}")
    
    # Try to connect
    cap = cv2.VideoCapture(camera_url)
    
    if not cap.isOpened():
        print("Failed to connect to IP camera")
        return
    
    print("Successfully connected to IP camera")
    
    # Try to read 10 frames
    for i in range(10):
        ret, frame = cap.read()
        if ret:
            print(f"Frame {i+1}: Successfully read frame: {frame.shape}")
            
            # Save first frame as test
            if i == 0:
                cv2.imwrite('ip_camera_test.jpg', frame)
                print("Saved test frame as 'ip_camera_test.jpg'")
        else:
            print(f"Frame {i+1}: Failed to read frame")
        
        time.sleep(0.5)
    
    cap.release()
    print("Test complete")

if __name__ == "__main__":
    # Replace with your phone's IP address and port
    IP_ADDRESS = "10.5.0.2:8080"  # Update this!
    test_ip_camera(IP_ADDRESS)