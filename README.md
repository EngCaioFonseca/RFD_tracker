# Rate of Force Development Tracker

A Python application for real-time tracking and analysis of barbell movements using computer vision. This tool helps athletes and coaches analyze lifting performance by measuring velocity, force, and power metrics during exercises.

## Features

- Real-time barbell tracking using computer vision
- Velocity, force, and power calculations
- Video recording capabilities
- Performance metrics visualization
- Historical data analysis
- Support for both webcam and IP camera inputs
- Exercise-specific analysis
- Data export capabilities

## Prerequisites

### System Requirements
- Python 3.8 or higher
- OpenCV-compatible camera (webcam or IP camera)
- Windows 11 with WSL2 (for WSL users) or Linux

### Required Packages

bash
pip install -r requirements.txt


## Installation

1. Clone the repository:
bash

git clone https://github.com/EngCaioFonseca/RFD_tracker.git

cd RFD_tracker

2. Install dependencies:

bash

pip install -r requirements.txt

3. Configure camera settings in `config.json` (created on first run)

## Camera Setup

### Using a Webcam
1. Ensure your webcam is connected and recognized by your system
   
2. Update `config.json` with:
json
{
"camera_type": "local",
"camera_index": 0
}

### Using IP Webcam (Phone Camera)
1. Install "IP Webcam" app from Google Play Store
2. Connect phone and computer to the same network
3. Start server in IP Webcam app
4. Note the IP address (e.g., http://192.168.1.100:8080)
5. Update `config.json` with:
json
{
"camera_type": "ip",
"ip_camera_address": "192.168.1.100:8080"
}

## Usage

1. Start the application:

bash
python RFD_tracker.py

2. Calibrate the system:
   - Click "Calibrate"
   - Place reference object (Olympic plate) in view
   - Follow on-screen instructions

3. Track a lift:
   - Select exercise type
   - Enter weight
   - Click "Start Tracking"
   - Perform the lift
   - Click "Stop Tracking"

4. View results:
   - Real-time metrics displayed on right panel
   - Historical data in "Lift History" tab
   - Analysis in "Analysis" tab

## Configuration

Edit `config.json` to customize:
- Camera settings
- Reference measurements
- Movement thresholds
- Output directory
- Video quality

Example configuration:
   
json
{
"camera_type": "ip",
"ip_camera_address": "192.168.1.100:8080",
"reference_height": 0.45,
"movement_threshold": 0.02,
"fps": 30,
"output_dir": "recordings",
"save_video": true,
"video_quality": 95
}

## Troubleshooting

### Camera Connection Issues
1. Check network connectivity (for IP camera)
2. Verify camera permissions
3. Test camera with:
bash

python test_camera.py

### WSL-Specific Issues
1. Ensure WSL2 is installed
2. Enable camera access in Windows settings
3. Configure `/etc/wsl.conf`:
ini
[boot]
systemd=true
[wsl2]
devices=true


## Data Analysis

The application provides:
- Velocity trends
- Force-velocity profiles
- Power development analysis
- Range of motion analysis
- Historical performance tracking

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenCV for computer vision capabilities
- PyQt6 for the GUI framework
- Ultralytics for YOLO implementation

## Support

For issues and feature requests, please use the GitHub issue tracker.

 


