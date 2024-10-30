"""
Rate of Force Development Tracker

A Python application that uses YOLO object detection and computer vision to track 
barbell movement and calculate advanced lifting metrics.

Features:
- YOLO-based barbell detection
- PyQt6-based GUI
- Video recording and playback
- Advanced metrics calculation
- Exercise recognition (optional)

Requirements:
- ultralytics (YOLO)
- PyQt6
- OpenCV
- NumPy
- torch

Author: Caio Fonseca
Date: 2024-10-30
"""

import sys
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import json
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, 
    QHBoxLayout, QPushButton, QLabel, QComboBox, 
    QSpinBox, QDoubleSpinBox, QGroupBox, QFormLayout,  # Added QFormLayout
    QTableWidget, QLineEdit, QTableWidgetItem, QFileDialog,
    QProgressBar, QTabWidget
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QThread, pyqtSignal



#######################
# Data Structures
#######################

@dataclass
class AdvancedLiftMetrics:
    """Store comprehensive metrics for a single lift"""
    # Basic metrics
    peak_velocity: float  # m/s
    mean_velocity: float  # m/s
    peak_power: float    # watts
    mean_power: float    # watts
    range_of_motion: float  # meters
    time_to_peak_velocity: float  # seconds
    
    # Advanced metrics
    acceleration: List[float]  # m/s²
    peak_acceleration: float   # m/s²
    force_curve: List[float]   # N
    peak_force: float         # N
    rate_of_force_dev: float  # N/s
    velocity_curve: List[float]  # m/s
    power_curve: List[float]    # watts
    
    # Metadata
    timestamp: datetime
    exercise_type: str
    weight_kg: float
    video_path: Optional[str]

class BarbellDetector:
    """YOLO-based barbell detection"""
    
    def __init__(self):
        """Initialize YOLO model"""
        self.model = YOLO('yolov8n.pt')  # Load pretrained model
        self.class_names = self.model.names
        self.barbell_class_id = 0  # Update based on your model
        
    def detect_barbell(self, frame):
        """
        Detect barbell in frame
        
        Args:
            frame: numpy array of image
            
        Returns:
            tuple: (x, y, confidence) of barbell center or None
        """
        results = self.model(frame)
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                if box.cls == self.barbell_class_id:
                    x1, y1, x2, y2 = box.xyxy[0]
                    confidence = box.conf[0]
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    return (center_x, center_y, confidence)
        
        return None

class VideoRecorder:
    """Handle video recording and saving"""
    
    def __init__(self, output_dir="recordings"):
        """Initialize video recorder"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.writer = None
        self.recording = False
        self.current_file = None
        
    def start_recording(self, frame_size, fps=30):
        """Start new recording"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_file = self.output_dir / f"lift_{timestamp}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(
            str(self.current_file), fourcc, fps, frame_size)
        self.recording = True
        
    def write_frame(self, frame):
        """Write frame to video"""
        if self.recording and self.writer:
            self.writer.write(frame)
            
    def stop_recording(self):
        """Stop recording and release writer"""
        if self.writer:
            self.writer.release()
        self.recording = False
        return self.current_file

class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced Barbell Tracker")
        self.setup_ui()
        
        # Initialize components
        self.detector = BarbellDetector()
        self.recorder = VideoRecorder()
        self.tracker = AdvancedBarbellTracker()
        
        # Setup video capture
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 30ms = ~33fps
        
        # State variables
        self.tracking = False
        self.recording = False
        
    def setup_ui(self):
        """Setup the user interface"""
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        
        # Left panel (video feed and controls)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Video feed
        self.video_label = QLabel()
        left_layout.addWidget(self.video_label)
        
        # Control buttons
        controls = QHBoxLayout()
        self.calibrate_btn = QPushButton("Calibrate")
        self.start_btn = QPushButton("Start Tracking")
        self.record_btn = QPushButton("Record")
        
        controls.addWidget(self.calibrate_btn)
        controls.addWidget(self.start_btn)
        controls.addWidget(self.record_btn)
        
        left_layout.addLayout(controls)
        
        # Right panel (metrics and settings)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Exercise settings
        settings_group = QWidget()
        settings_layout = QVBoxLayout(settings_group)
        
        # Exercise type selector
        exercise_layout = QHBoxLayout()
        exercise_layout.addWidget(QLabel("Exercise:"))
        self.exercise_combo = QComboBox()
        self.exercise_combo.addItems([
            "Squat", "Bench Press", "Deadlift", "Clean", "Snatch"
        ])
        exercise_layout.addWidget(self.exercise_combo)
        settings_layout.addLayout(exercise_layout)
        
        # Weight input
        weight_layout = QHBoxLayout()
        weight_layout.addWidget(QLabel("Weight (kg):"))
        self.weight_spin = QDoubleSpinBox()
        self.weight_spin.setRange(0, 500)
        self.weight_spin.setValue(100)
        weight_layout.addWidget(self.weight_spin)
        settings_layout.addLayout(weight_layout)
        
        right_layout.addWidget(settings_group)
        
        # Metrics display
        self.metrics_label = QLabel("No metrics available")
        self.metrics_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        right_layout.addWidget(self.metrics_label)
        
        # Add panels to main layout
        layout.addWidget(left_panel, stretch=2)
        layout.addWidget(right_panel, stretch=1)
        
        # Connect signals
        self.calibrate_btn.clicked.connect(self.calibrate)
        self.start_btn.clicked.connect(self.toggle_tracking)
        self.record_btn.clicked.connect(self.toggle_recording)
        
    def update_frame(self):
        """Update video frame and process tracking"""
        ret, frame = self.cap.read()
        if not ret:
            return
            
        # Process frame
        if self.tracking:
            # Detect barbell
            barbell_pos = self.detector.detect_barbell(frame)
            if barbell_pos:
                x, y, conf = barbell_pos
                # Draw detection
                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
                # Update tracking
                self.tracker.update_position(x, y)
                # Update metrics display
                self.update_metrics()
        
        # Record frame if needed
        if self.recording:
            self.recorder.write_frame(frame)
        
        # Convert frame to Qt format and display
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))


class AdvancedBarbellTracker:
    """Advanced barbell tracking and metrics calculation"""
    
    def __init__(self):
        """Initialize tracker"""
        # Position tracking
        self.positions = []  # List of (x, y) tuples
        self.timestamps = []  # List of timestamps
        self.calibration_factor = None  # pixels to meters
        
        # Calculated metrics
        self.velocities = []
        self.accelerations = []
        self.forces = []
        self.power_values = []
        
        # Movement detection
        self.movement_started = False
        self.movement_threshold = 0.02  # meters
        self.rep_count = 0
        
        # Exercise recognition
        self.movement_patterns = []
        self.exercise_classifier = ExerciseClassifier()
        
    def calibrate(self, reference_height_meters=0.45):
        """Calibrate using known reference height (e.g., plate diameter)"""
        if len(self.positions) < 2:
            return False
            
        pixel_height = abs(self.positions[-1][1] - self.positions[0][1])
        self.calibration_factor = reference_height_meters / pixel_height
        return True
        
    def update_position(self, x, y, timestamp=None):
        """Update position and calculate metrics"""
        if timestamp is None:
            timestamp = datetime.now().timestamp()
            
        self.positions.append((x, y))
        self.timestamps.append(timestamp)
        
        if len(self.positions) >= 2:
            self._calculate_metrics()
            self._detect_movement()
            self._update_movement_pattern()
            
    def _calculate_metrics(self):
        """Calculate comprehensive movement metrics"""
        # Get last two positions and timestamps
        p1, p2 = self.positions[-2:]
        t1, t2 = self.timestamps[-2:]
        dt = t2 - t1
        
        # Calculate displacement in meters
        dx = (p2[0] - p1[0]) * self.calibration_factor
        dy = (p2[1] - p1[1]) * self.calibration_factor
        displacement = np.sqrt(dx**2 + dy**2)
        
        # Calculate velocity
        velocity = displacement / dt
        self.velocities.append(velocity)
        
        # Calculate acceleration
        if len(self.velocities) >= 2:
            acceleration = (self.velocities[-1] - self.velocities[-2]) / dt
            self.accelerations.append(acceleration)
        
        # Calculate force and power if weight is set
        if hasattr(self, 'weight_kg'):
            force = self.weight_kg * 9.81 + self.weight_kg * (
                self.accelerations[-1] if self.accelerations else 0)
            self.forces.append(force)
            
            power = force * velocity
            self.power_values.append(power)
            
    def _detect_movement(self):
        """Detect start and end of movement"""
        if not self.movement_started:
            # Check for movement start
            if len(self.velocities) >= 3 and all(v > self.movement_threshold 
                                               for v in self.velocities[-3:]):
                self.movement_started = True
                self.movement_patterns = []  # Reset pattern for new rep
        else:
            # Check for movement end
            if len(self.velocities) >= 3 and all(v < self.movement_threshold 
                                               for v in self.velocities[-3:]):
                self.movement_started = False
                self.rep_count += 1
                self._classify_exercise()
                
    def _update_movement_pattern(self):
        """Update movement pattern for exercise recognition"""
        if self.movement_started and len(self.positions) >= 2:
            # Calculate movement direction and velocity
            p1, p2 = self.positions[-2:]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            velocity = self.velocities[-1]
            
            # Create pattern feature vector
            pattern = [dx, dy, velocity]
            self.movement_patterns.append(pattern)
            
    def get_current_metrics(self) -> AdvancedLiftMetrics:
        """Get current lift metrics"""
        if not self.velocities:
            return None
            
        return AdvancedLiftMetrics(
            peak_velocity=max(self.velocities),
            mean_velocity=np.mean(self.velocities),
            peak_power=max(self.power_values) if self.power_values else 0,
            mean_power=np.mean(self.power_values) if self.power_values else 0,
            range_of_motion=self._calculate_rom(),
            time_to_peak_velocity=self._calculate_time_to_peak(),
            acceleration=self.accelerations,
            peak_acceleration=max(self.accelerations) if self.accelerations else 0,
            force_curve=self.forces,
            peak_force=max(self.forces) if self.forces else 0,
            rate_of_force_dev=self._calculate_rfd(),
            velocity_curve=self.velocities,
            power_curve=self.power_values,
            timestamp=datetime.now(),
            exercise_type=self.exercise_classifier.current_exercise,
            weight_kg=getattr(self, 'weight_kg', 0),
            video_path=None
        )
        
    def _calculate_rom(self):
        """Calculate range of motion"""
        if not self.positions:
            return 0
            
        y_positions = [p[1] for p in self.positions]
        rom = (max(y_positions) - min(y_positions)) * self.calibration_factor
        return abs(rom)
        
    def _calculate_time_to_peak(self):
        """Calculate time to peak velocity"""
        if not self.velocities:
            return 0
            
        peak_idx = np.argmax(self.velocities)
        return self.timestamps[peak_idx] - self.timestamps[0]
        
    def _calculate_rfd(self):
        """Calculate rate of force development"""
        if not self.forces or len(self.forces) < 2:
            return 0
            
        peak_force_idx = np.argmax(self.forces)
        time_to_peak = self.timestamps[peak_force_idx] - self.timestamps[0]
        if time_to_peak == 0:
            return 0
            
        return self.forces[peak_force_idx] / time_to_peak

class ExerciseClassifier:
    """Exercise recognition using movement patterns"""
    
    def __init__(self):
        """Initialize exercise classifier"""
        self.current_exercise = None
        self.movement_patterns = {
            'Squat': {'vertical_dominant': True, 'velocity_profile': 'symmetric'},
            'Bench Press': {'vertical_dominant': False, 'velocity_profile': 'symmetric'},
            'Deadlift': {'vertical_dominant': True, 'velocity_profile': 'asymmetric'},
            'Clean': {'vertical_dominant': True, 'velocity_profile': 'explosive'},
            'Snatch': {'vertical_dominant': True, 'velocity_profile': 'explosive_extended'}
        }
        
    def classify_movement(self, patterns):
        """
        Classify exercise based on movement patterns
        
        Args:
            patterns: List of [dx, dy, velocity] measurements
        """
        if not patterns:
            return None
            
        # Extract pattern features
        vertical_movement = sum(abs(p[1]) for p in patterns)
        horizontal_movement = sum(abs(p[0]) for p in patterns)
        peak_velocity = max(p[2] for p in patterns)
        velocity_profile = self._analyze_velocity_profile(patterns)
        
        # Determine movement characteristics
        vertical_dominant = vertical_movement > (2 * horizontal_movement)
        
        # Match against known patterns
        scores = {}
        for exercise, pattern in self.movement_patterns.items():
            score = self._calculate_pattern_match(
                pattern,
                {
                    'vertical_dominant': vertical_dominant,
                    'velocity_profile': velocity_profile
                }
            )
            scores[exercise] = score
            
        # Select best match
        self.current_exercise = max(scores.items(), key=lambda x: x[1])[0]
        return self.current_exercise
        
    def _analyze_velocity_profile(self, patterns):
        """Analyze velocity profile characteristics"""
        velocities = [p[2] for p in patterns]
        
        # Calculate profile characteristics
        peak_idx = np.argmax(velocities)
        time_to_peak = peak_idx / len(velocities)
        peak_velocity = max(velocities)
        mean_velocity = np.mean(velocities)
        
        # Classify profile
        if peak_velocity > (3 * mean_velocity):
            return 'explosive'
        elif time_to_peak < 0.3:
            return 'explosive_extended'
        elif abs(time_to_peak - 0.5) < 0.1:
            return 'symmetric'
        else:
            return 'asymmetric'
            
    def _calculate_pattern_match(self, pattern, observed):
        """Calculate how well observed patterns match known patterns"""
        score = 0
        if pattern['vertical_dominant'] == observed['vertical_dominant']:
            score += 1
        if pattern['velocity_profile'] == observed['velocity_profile']:
            score += 2
        return score


"""
Video Recording and Analysis Components
"""

class VideoManager:
    """Handle video recording, playback, and analysis"""
    
    def __init__(self, output_dir="recordings"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Recording components
        self.writer = None
        self.recording = False
        self.current_file = None
        
        # Playback components
        self.playback_video = None
        self.playback_frame_count = 0
        self.current_frame = 0
        
        # Analysis data
        self.frame_timestamps = []
        self.frame_metrics = []
        
    def start_recording(self, frame_size, fps=30):
        """Start new video recording"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_file = self.output_dir / f"lift_{timestamp}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(
            str(self.current_file), fourcc, fps, frame_size)
        self.recording = True
        
    def write_frame(self, frame, metrics=None):
        """Write frame and associated metrics"""
        if self.recording and self.writer:
            self.writer.write(frame)
            self.frame_timestamps.append(datetime.now())
            self.frame_metrics.append(metrics)
            
    def stop_recording(self):
        """Stop recording and save metadata"""
        if self.writer:
            self.writer.release()
            self.save_metadata()
        self.recording = False
        return self.current_file
        
    def save_metadata(self):
        """Save frame timestamps and metrics"""
        metadata = {
            'timestamps': [ts.isoformat() for ts in self.frame_timestamps],
            'metrics': self.frame_metrics
        }
        metadata_file = self.current_file.with_suffix('.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)
            
    def load_video(self, video_path):
        """Load video for playback and analysis"""
        self.playback_video = cv2.VideoCapture(str(video_path))
        self.playback_frame_count = int(self.playback_video.get(
            cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0
        
        # Load associated metadata
        metadata_file = Path(video_path).with_suffix('.json')
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                self.frame_timestamps = [parser.parse(ts) 
                                      for ts in metadata['timestamps']]
                self.frame_metrics = metadata['metrics']
                
    def get_next_frame(self):
        """Get next frame for playback"""
        if self.playback_video and self.current_frame < self.playback_frame_count:
            ret, frame = self.playback_video.read()
            if ret:
                self.current_frame += 1
                return frame, self.get_frame_metrics()
        return None, None
        
    def get_frame_metrics(self):
        """Get metrics for current frame"""
        if self.frame_metrics and self.current_frame < len(self.frame_metrics):
            return self.frame_metrics[self.current_frame]
        return None

class DataVisualizer:
    """Handle real-time and post-analysis visualization"""
    
    def __init__(self):
        self.figure = plt.figure(figsize=(12, 8))
        self.velocity_ax = None
        self.force_ax = None
        self.power_ax = None
        self.setup_plots()
        
    def setup_plots(self):
        """Initialize plot layout"""
        gs = self.figure.add_gridspec(3, 1)
        self.velocity_ax = self.figure.add_subplot(gs[0])
        self.force_ax = self.figure.add_subplot(gs[1])
        self.power_ax = self.figure.add_subplot(gs[2])
        
        self.velocity_ax.set_title('Velocity Profile')
        self.force_ax.set_title('Force Profile')
        self.power_ax.set_title('Power Profile')
        
        plt.tight_layout()
        
    def update_plots(self, metrics: AdvancedLiftMetrics):
        """Update real-time plots with new metrics"""
        self._plot_curve(self.velocity_ax, metrics.velocity_curve, 'Velocity (m/s)')
        self._plot_curve(self.force_ax, metrics.force_curve, 'Force (N)')
        self._plot_curve(self.power_ax, metrics.power_curve, 'Power (W)')
        plt.draw()
        
    def _plot_curve(self, ax, data, ylabel):
        """Plot single curve with styling"""
        ax.clear()
        ax.plot(data, 'b-', linewidth=2)
        ax.set_ylabel(ylabel)
        ax.grid(True)
        
    def generate_report(self, metrics: AdvancedLiftMetrics):
        """Generate comprehensive performance report"""
        fig = plt.figure(figsize=(15, 10))
        
        # Metrics summary
        plt.subplot(2, 2, 1)
        self._plot_metrics_summary(metrics)
        
        # Velocity profile
        plt.subplot(2, 2, 2)
        self._plot_velocity_profile(metrics)
        
        # Force-velocity curve
        plt.subplot(2, 2, 3)
        self._plot_force_velocity(metrics)
        
        # Power curve
        plt.subplot(2, 2, 4)
        self._plot_power_curve(metrics)
        
        plt.tight_layout()
        return fig
        
    def _plot_metrics_summary(self, metrics):
        """Plot summary of key metrics"""
        labels = ['Peak Velocity', 'Mean Velocity', 'Peak Force', 'Peak Power']
        values = [metrics.peak_velocity, metrics.mean_velocity,
                 metrics.peak_force/1000, metrics.peak_power/1000]
        
        plt.bar(labels, values)
        plt.title('Performance Summary')
        plt.xticks(rotation=45)
        
    def _plot_velocity_profile(self, metrics):
        """Plot detailed velocity profile"""
        plt.plot(metrics.velocity_curve)
        plt.title('Velocity Profile')
        plt.xlabel('Time (frames)')
        plt.ylabel('Velocity (m/s)')
        
    def _plot_force_velocity(self, metrics):
        """Plot force-velocity relationship"""
        plt.scatter(metrics.velocity_curve, metrics.force_curve)
        plt.title('Force-Velocity Curve')
        plt.xlabel('Velocity (m/s)')
        plt.ylabel('Force (N)')
        
    def _plot_power_curve(self, metrics):
        """Plot power output curve"""
        plt.plot(metrics.power_curve)
        plt.title('Power Output')
        plt.xlabel('Time (frames)')
        plt.ylabel('Power (W)')

class PerformanceAnalyzer:
    """Analyze lift performance and generate insights"""
    
    def __init__(self):
        self.lift_history = []
        
    def add_lift(self, metrics: AdvancedLiftMetrics):
        """Add new lift metrics to history"""
        self.lift_history.append(metrics)
        
    def analyze_trends(self):
        """Analyze performance trends over time"""
        if not self.lift_history:
            return None
            
        trends = {
            'peak_velocity': self._calculate_trend('peak_velocity'),
            'peak_power': self._calculate_trend('peak_power'),
            'rate_of_force_dev': self._calculate_trend('rate_of_force_dev')
        }
        
        return trends
        
    def _calculate_trend(self, metric_name):
        """Calculate trend for specific metric"""
        values = [getattr(m, metric_name) for m in self.lift_history]
        times = range(len(values))
        
        if len(values) < 2:
            return 0
            
        slope, _, r_value, _, _ = stats.linregress(times, values)
        return {
            'slope': slope,
            'r_squared': r_value**2
        }
        
    def generate_insights(self):
        """Generate performance insights"""
        trends = self.analyze_trends()
        if not trends:
            return "Not enough data for analysis"
            
        insights = []
        
        # Velocity trends
        vel_trend = trends['peak_velocity']['slope']
        if vel_trend > 0:
            insights.append("Velocity is improving over time")
        elif vel_trend < 0:
            insights.append("Velocity shows declining trend")
            
        # Power development
        power_trend = trends['peak_power']['slope']
        if power_trend > 0:
            insights.append("Power output is increasing")
        
        # Force development
        rfd_trend = trends['rate_of_force_dev']['slope']
        if rfd_trend > 0:
            insights.append("Rate of force development is improving")
        
        return "\n".join(insights)

"""
Database, Advanced GUI, and Data Management Components
"""

import sqlite3
from PyQt6.QtWidgets import (QTabWidget, QTableWidget, QTableWidgetItem, 
                            QFileDialog, QProgressBar, QGroupBox)
from PyQt6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis
import pandas as pd
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import seaborn as sns

class DatabaseManager:
    """Handle database operations for lift tracking"""
    
    def __init__(self, db_path="lift_tracker.db"):
        """Initialize database connection"""
        self.db_path = db_path
        self.setup_database()
        
    def setup_database(self):
        """Create database tables if they don't exist"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create lifts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS lifts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    exercise_type TEXT,
                    weight_kg REAL,
                    peak_velocity REAL,
                    mean_velocity REAL,
                    peak_power REAL,
                    mean_power REAL,
                    range_of_motion REAL,
                    time_to_peak_velocity REAL,
                    peak_force REAL,
                    rate_of_force_dev REAL,
                    video_path TEXT
                )
            ''')
            
            # Create metrics table for detailed data
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS lift_metrics (
                    lift_id INTEGER,
                    frame_number INTEGER,
                    velocity REAL,
                    force REAL,
                    power REAL,
                    FOREIGN KEY (lift_id) REFERENCES lifts (id)
                )
            ''')
            
    def save_lift(self, metrics: AdvancedLiftMetrics) -> int:
        """Save lift metrics to database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Insert main lift data
            cursor.execute('''
                INSERT INTO lifts (
                    timestamp, exercise_type, weight_kg, peak_velocity,
                    mean_velocity, peak_power, mean_power, range_of_motion,
                    time_to_peak_velocity, peak_force, rate_of_force_dev,
                    video_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metrics.timestamp, metrics.exercise_type, metrics.weight_kg,
                metrics.peak_velocity, metrics.mean_velocity, metrics.peak_power,
                metrics.mean_power, metrics.range_of_motion,
                metrics.time_to_peak_velocity, metrics.peak_force,
                metrics.rate_of_force_dev, metrics.video_path
            ))
            
            lift_id = cursor.lastrowid
            
            # Insert detailed metrics
            for i in range(len(metrics.velocity_curve)):
                cursor.execute('''
                    INSERT INTO lift_metrics (
                        lift_id, frame_number, velocity, force, power
                    ) VALUES (?, ?, ?, ?, ?)
                ''', (
                    lift_id, i, metrics.velocity_curve[i],
                    metrics.force_curve[i], metrics.power_curve[i]
                ))
                
            return lift_id
            
    def get_lift_history(self, exercise_type=None, date_range=None) -> pd.DataFrame:
        """Retrieve lift history with optional filters"""
        query = "SELECT * FROM lifts"
        params = []
        
        if exercise_type or date_range:
            query += " WHERE"
            conditions = []
            
            if exercise_type:
                conditions.append("exercise_type = ?")
                params.append(exercise_type)
                
            if date_range:
                start_date, end_date = date_range
                conditions.append("timestamp BETWEEN ? AND ?")
                params.extend([start_date, end_date])
                
            query += " AND ".join(conditions)
            
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=params)

class VideoProcessingThread(QThread):
    """Worker thread for video processing"""
    progress = pyqtSignal(int)
    frame_processed = pyqtSignal(object)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, video_path, exercise_type, weight):
        super().__init__()
        self.video_path = video_path
        self.exercise_type = exercise_type
        self.weight = weight
        self.model = YOLO('yolov8n.pt')
    
    def run(self):
        try:
            cap = cv2.VideoCapture(self.video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            processed_frames = 0
            
            # Data storage
            positions = []
            timestamps = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect barbell
                results = self.model(frame)
                
                # Process detections
                for result in results:
                    for box, cls, conf in zip(result.boxes.xyxy, 
                                            result.boxes.cls, 
                                            result.boxes.conf):
                        if cls == 1 and conf > 0.5:  # Assuming class 1 is barbell
                            x1, y1, x2, y2 = box.tolist()
                            center_x = (x1 + x2) / 2
                            center_y = (y1 + y2) / 2
                            
                            # Store position
                            positions.append((center_x, center_y))
                            timestamps.append(processed_frames / cap.get(cv2.CAP_PROP_FPS))
                            
                            # Draw detection
                            cv2.circle(frame, (int(center_x), int(center_y)), 
                                     5, (0, 255, 0), -1)
                
                # Emit processed frame
                self.frame_processed.emit(frame)
                
                # Update progress
                processed_frames += 1
                progress = int((processed_frames / total_frames) * 100)
                self.progress.emit(progress)
            
            cap.release()
            
            # Calculate metrics
            metrics = self.calculate_metrics(positions, timestamps)
            self.finished.emit(metrics)
            
        except Exception as e:
            self.error.emit(str(e))
    
    def calculate_metrics(self, positions, timestamps):
        """Calculate performance metrics from positions and timestamps"""
        # Implementation similar to live tracking metrics calculation
        # Return dictionary of metrics
        return {
            'exercise_type': self.exercise_type,
            'weight_kg': self.weight,
            'timestamps': timestamps,
            'positions': positions,
            # Add other metrics
        }

class AdvancedGUI(QMainWindow):
    """Enhanced GUI with advanced features for barbell tracking"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Barbell Velocity Tracker")
        
        # Initialize components
        self.setup_logging()
        self.load_config()
        self.init_tracking_system()
        self.setup_advanced_ui()
        self.setup_camera()
        
        # State variables
        self.tracking = False
        self.recording = False
        self.calibration_mode = False
        self.calibration_factor = None  # pixels to meters
        
        self.logger.info("Application initialized")

    def setup_logging(self):
        """Setup logging configuration"""
        self.logger = logging.getLogger('BarbellTracker')
        self.logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler('barbell_tracker.log')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def load_config(self):
        """Load configuration from file"""
        default_config = {
        'camera_index': 0,
        'reference_height': 0.45,  # meters (Olympic plate diameter)
        'movement_threshold': 0.02,  # meters
        'fps': 30,
        'output_dir': 'recordings',
        'min_confidence': 0.5,
        'calibration_reference': 0.45,  # meters
            'save_video': True,
            'video_quality': 95
        }
    
        try:
            with open('config.json', 'r') as f:
                loaded_config = json.load(f)
            # Merge loaded config with defaults
            self.config = {**default_config, **loaded_config}
        except FileNotFoundError:
            # Use defaults if no config file exists
            self.config = default_config
            self.save_config()
        except Exception as e:
            self.logger.error(f"Error loading config: {str(e)}")
            self.config = default_config
        
        self.logger.info("Configuration loaded successfully")

    def save_config(self):
        """Save configuration to file"""
        with open('config.json', 'w') as f:
            json.dump(self.config, f, indent=4)

    def init_tracking_system(self):
        """Initialize tracking system components"""
        # Initialize YOLO model for barbell detection
        try:
            self.model = YOLO('yolov8n.pt')
            self.logger.info("YOLO model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {str(e)}")
            self.model = None

        # Initialize data structures for tracking
        self.current_lift = {
            'positions': [],
            'timestamps': [],
            'velocities': [],
            'forces': [],
            'power': []
        }

    def setup_advanced_ui(self):
        """Setup the advanced user interface"""
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # Create tab widget
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        # Add tabs
        self.tabs.addTab(self.create_tracking_tab(), "Live Tracking")
        self.tabs.addTab(self.create_history_tab(), "Lift History")
        self.tabs.addTab(self.create_analysis_tab(), "Analysis")
        self.tabs.addTab(self.create_settings_tab(), "Settings")
        self.tabs.addTab(self.create_video_processing_tab(), "Video Processing")
        
        # Create status bar
        self.statusBar().showMessage("Ready")
        
        # Set window size
        self.setMinimumSize(1200, 800)

    def create_tracking_tab(self):
        """Create the live tracking interface tab"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        
        # Add tracking panel
        left_panel = self.create_tracking_panel()
        right_panel = self.create_metrics_panel()
        
        layout.addWidget(left_panel, stretch=2)
        layout.addWidget(right_panel, stretch=1)
        
        return widget

    def create_tracking_panel(self):
        """Create panel for video feed and tracking controls"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Video feed
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        layout.addWidget(self.video_label)
        
        # Control buttons
        controls = QHBoxLayout()
        
        self.calibrate_btn = QPushButton("Calibrate")
        self.start_btn = QPushButton("Start Tracking")
        self.record_btn = QPushButton("Record")
        
        controls.addWidget(self.calibrate_btn)
        controls.addWidget(self.start_btn)
        controls.addWidget(self.record_btn)
        
        # Exercise settings
        settings = QHBoxLayout()
        
        settings.addWidget(QLabel("Exercise:"))
        self.exercise_combo = QComboBox()
        self.exercise_combo.addItems([
            "Squat", "Bench Press", "Deadlift", "Clean", "Snatch"
        ])
        settings.addWidget(self.exercise_combo)
        
        settings.addWidget(QLabel("Weight (kg):"))
        self.weight_spin = QDoubleSpinBox()
        self.weight_spin.setRange(0, 500)
        self.weight_spin.setValue(100)
        settings.addWidget(self.weight_spin)
        
        # Add layouts to panel
        layout.addLayout(controls)
        layout.addLayout(settings)
        
        # Connect signals
        self.calibrate_btn.clicked.connect(self.on_calibrate)
        self.start_btn.clicked.connect(self.on_start_tracking)
        self.record_btn.clicked.connect(self.on_record)
        
        return panel
    
    def create_metrics_panel(self):
        """Create panel for real-time metrics display"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Current metrics display
        metrics_group = QGroupBox("Current Metrics")
        metrics_layout = QVBoxLayout(metrics_group)
        
        self.velocity_label = QLabel("Velocity: 0.00 m/s")
        self.power_label = QLabel("Power: 0.00 W")
        self.force_label = QLabel("Force: 0.00 N")
        self.rom_label = QLabel("ROM: 0.00 m")
        self.rfd_label = QLabel("RFD: 0.00 N/s")
        
        metrics_layout.addWidget(self.velocity_label)
        metrics_layout.addWidget(self.power_label)
        metrics_layout.addWidget(self.force_label)
        metrics_layout.addWidget(self.rom_label)
        metrics_layout.addWidget(self.rfd_label)
        
        metrics_group.setLayout(metrics_layout)
        layout.addWidget(metrics_group)
        
        # Real-time plots
        plots_group = QGroupBox("Real-time Plots")
        plots_layout = QVBoxLayout(plots_group)
        
        self.figure = plt.figure(figsize=(6, 8))
        self.canvas = FigureCanvas(self.figure)
        plots_layout.addWidget(self.canvas)
        
        layout.addWidget(plots_group)
        
        return panel

    def create_history_tab(self):
        """Create the lift history interface tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Filters
        filters = QHBoxLayout()
        
        filters.addWidget(QLabel("Exercise:"))
        self.history_exercise_filter = QComboBox()
        self.history_exercise_filter.addItems(["All"] + [
            "Squat", "Bench Press", "Deadlift", "Clean", "Snatch"
        ])
        filters.addWidget(self.history_exercise_filter)
        
        layout.addLayout(filters)
        
        # History table
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(8)
        self.history_table.setHorizontalHeaderLabels([
            "Date", "Exercise", "Weight (kg)", "Peak Velocity (m/s)",
            "Mean Velocity (m/s)", "Peak Power (W)", "Peak Force (N)", "RFD (N/s)"
        ])
        layout.addWidget(self.history_table)
        
        # Connect signals
        self.history_exercise_filter.currentTextChanged.connect(self.update_history_table)
        
        return widget

    def create_analysis_tab(self):
        """Create the analysis interface tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Analysis charts
        charts_group = QGroupBox("Performance Analysis")
        charts_layout = QVBoxLayout(charts_group)
        
        self.analysis_figure = plt.figure(figsize=(10, 8))
        self.analysis_canvas = FigureCanvas(self.analysis_figure)
        charts_layout.addWidget(self.analysis_canvas)
        
        layout.addWidget(charts_group)
        
        # Analysis controls
        controls = QHBoxLayout()
        
        self.analysis_type = QComboBox()
        self.analysis_type.addItems([
            "Velocity Trends", "Force-Velocity Profile", 
            "Power Development", "ROM Analysis"
        ])
        controls.addWidget(self.analysis_type)
        
        update_btn = QPushButton("Update Analysis")
        update_btn.clicked.connect(self.update_analysis)
        controls.addWidget(update_btn)
        
        layout.addLayout(controls)
        
        return widget

    def create_settings_tab(self):
        """Create the settings interface tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Camera settings
        camera_group = QGroupBox("Camera Settings")
        camera_layout = QFormLayout()
        
        self.camera_select = QComboBox()
        self.camera_select.addItems(["Default Camera", "IP Camera"])
        
        self.ip_address = QLineEdit()
        self.ip_address.setPlaceholderText("Enter IP camera address")
        
        camera_layout.addRow("Camera:", self.camera_select)
        camera_layout.addRow("IP Address:", self.ip_address)
        camera_group.setLayout(camera_layout)
        
        layout.addWidget(camera_group)
        
        # Tracking settings
        tracking_group = QGroupBox("Tracking Settings")
        tracking_layout = QFormLayout()
        
        self.reference_height = QDoubleSpinBox()
        self.reference_height.setValue(self.config['reference_height'])
        self.reference_height.setSingleStep(0.01)
        
        self.movement_threshold = QDoubleSpinBox()
        self.movement_threshold.setValue(self.config['movement_threshold'])
        self.movement_threshold.setSingleStep(0.01)
        
        tracking_layout.addRow("Reference Height (m):", self.reference_height)
        tracking_layout.addRow("Movement Threshold (m):", self.movement_threshold)
        tracking_group.setLayout(tracking_layout)
        
        layout.addWidget(tracking_group)
        
        # Save button
        save_btn = QPushButton("Save Settings")
        save_btn.clicked.connect(self.save_settings)
        layout.addWidget(save_btn)
        
        return widget

    def setup_camera(self):
        """Setup IP camera connection"""
        try:
            # Get IP camera URL from settings or use default
            ip_address = self.ip_address.text() if hasattr(self, 'ip_address') else "10.5.0.2:8080"
            self.camera_url = f"http://{ip_address}/video"
            
            # Try to connect to IP camera
            self.cap = cv2.VideoCapture(self.camera_url)
            
            if not self.cap.isOpened():
                self.logger.error(f"Failed to connect to IP camera at {self.camera_url}")
                self.statusBar().showMessage(
                    "Failed to connect to IP camera. Please check the IP address and ensure the app is running.")
                return
                
            # Setup frame update timer
            self.timer = QTimer()
            self.timer.timeout.connect(self.update_frame)
            self.timer.start(int(1000/self.config['fps']))
            
            self.logger.info(f"IP camera connected successfully at {self.camera_url}")
            self.statusBar().showMessage("IP camera connected successfully")
        except Exception as e:
            self.logger.error(f"Camera setup failed: {str(e)}")
            self.statusBar().showMessage(f"Camera setup failed: {str(e)}")


    def detect_barbell(self, frame):
        """Detect barbell in frame using YOLO"""
        try:
            if self.model is None:
                return None
                
            results = self.model(frame)
            
            # Filter for barbell detections
            for result in results:
                for box, cls, conf in zip(result.boxes.xyxy, 
                                        result.boxes.cls, 
                                        result.boxes.conf):
                    if cls == 1 and conf > 0.5:  # Assuming class 1 is barbell
                        x1, y1, x2, y2 = box.tolist()
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        return (center_x, center_y, conf)
            
            return None
        except Exception as e:
            self.logger.error(f"Barbell detection error: {str(e)}")
            return None

    def calculate_metrics(self, position, timestamp):
        """Calculate movement metrics"""
        try:
            if not self.current_lift['positions']:
                self.current_lift['positions'].append(position)
                self.current_lift['timestamps'].append(timestamp)
                return
                
            # Calculate displacement
            prev_pos = self.current_lift['positions'][-1]
            dx = position[0] - prev_pos[0]
            dy = position[1] - prev_pos[1]
            displacement = np.sqrt(dx**2 + dy**2) * self.calibration_factor
            
            # Calculate time difference
            dt = timestamp - self.current_lift['timestamps'][-1]
            
            # Calculate velocity
            velocity = displacement / dt if dt > 0 else 0
            
            # Calculate force (F = ma)
            weight = self.weight_spin.value()
            force = weight * (9.81 + velocity)  # Include acceleration due to gravity
            
            # Calculate power (P = F * v)
            power = force * velocity
            
            # Update lists
            self.current_lift['positions'].append(position)
            self.current_lift['timestamps'].append(timestamp)
            self.current_lift['velocities'].append(velocity)
            self.current_lift['forces'].append(force)
            self.current_lift['power'].append(power)
            
            return velocity, force, power
            
        except Exception as e:
            self.logger.error(f"Metrics calculation error: {str(e)}")
            return None, None, None

    def update_frame(self):
        """Update video frame and process tracking"""
        try:
            ret, frame = self.cap.read()
            if not ret:
                return
            
            if self.tracking:
                # Detect barbell
                detection = self.detect_barbell(frame)
                if detection:
                    x, y, conf = detection
                    
                    # Draw detection
                    cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
                    
                    # Calculate metrics
                    timestamp = datetime.now().timestamp()
                    metrics = self.calculate_metrics((x, y), timestamp)
                    
                    if metrics:
                        velocity, force, power = metrics
                        self.update_metrics_display(velocity, force, power)
            
            if self.recording and hasattr(self, 'video_writer'):
                self.video_writer.write(frame)
            
            # Convert frame to Qt format and display
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, 
                            QImage.Format.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qt_image))
            
        except Exception as e:
            self.logger.error(f"Frame update error: {str(e)}")

    def on_calibrate(self):
        """Handle calibration button click"""
        try:
            if not self.calibration_mode:
                self.calibration_mode = True
                self.calibrate_btn.setText("Set Reference")
                self.statusBar().showMessage(
                    "Place reference object (Olympic plate) in frame and click Set Reference")
                self.reference_points = []
            else:
                # Get current frame
                ret, frame = self.cap.read()
                if not ret:
                    return
                
                # Detect barbell/plate
                detection = self.detect_barbell(frame)
                if detection:
                    x, y, conf = detection
                    self.reference_points.append((x, y))
                    
                    if len(self.reference_points) == 2:
                        # Calculate calibration factor
                        p1, p2 = self.reference_points
                        pixel_height = abs(p2[1] - p1[1])
                        self.calibration_factor = (
                            self.config['reference_height'] / pixel_height)
                        
                        self.calibration_mode = False
                        self.calibrate_btn.setText("Calibrate")
                        self.statusBar().showMessage(
                            f"Calibration complete: {self.calibration_factor:.4f} m/pixel")
                        self.reference_points = []
                    else:
                        self.statusBar().showMessage(
                            "Now place reference at second position")
                else:
                    self.statusBar().showMessage("No reference object detected")
                    
        except Exception as e:
            self.logger.error(f"Calibration error: {str(e)}")
            self.statusBar().showMessage(f"Calibration error: {str(e)}")

    def on_start_tracking(self):
        """Handle start tracking button click"""
        try:
            if not self.calibration_factor:
                self.statusBar().showMessage("Please calibrate first")
                return
                
            if not self.tracking:
                # Start tracking
                self.tracking = True
                self.start_btn.setText("Stop Tracking")
                self.statusBar().showMessage("Tracking started")
                
                # Initialize tracking data
                self.current_lift = {
                    'positions': [],
                    'timestamps': [],
                    'velocities': [],
                    'forces': [],
                    'power': []
                }
                
                # Clear plots
                self.figure.clear()
                self.canvas.draw()
            else:
                # Stop tracking
                self.tracking = False
                self.start_btn.setText("Start Tracking")
                self.statusBar().showMessage("Tracking stopped")
                
                # Process and save lift data
                if self.current_lift['positions']:
                    self.process_lift_data()
                    
        except Exception as e:
            self.logger.error(f"Tracking error: {str(e)}")
            self.statusBar().showMessage(f"Tracking error: {str(e)}")

    def on_record(self):
        """Handle record button click"""
        try:
            if not self.recording:
                # Start recording
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = Path(self.config['output_dir']) / f"lift_{timestamp}.mp4"
                
                # Ensure output directory exists
                filename.parent.mkdir(exist_ok=True)
                
                # Initialize video writer
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                frame_size = (int(self.cap.get(3)), int(self.cap.get(4)))
                self.video_writer = cv2.VideoWriter(
                    str(filename), fourcc, self.config['fps'], frame_size)
                
                self.recording = True
                self.record_btn.setText("Stop Recording")
                self.statusBar().showMessage("Recording started")
            else:
                # Stop recording
                self.recording = False
                self.record_btn.setText("Record")
                
                if hasattr(self, 'video_writer'):
                    self.video_writer.release()
                    delattr(self, 'video_writer')
                
                self.statusBar().showMessage("Recording stopped")
                
        except Exception as e:
            self.logger.error(f"Recording error: {str(e)}")
            self.statusBar().showMessage(f"Recording error: {str(e)}")

    def update_metrics_display(self, velocity, force, power):
        """Update real-time metrics display"""
        try:
            # Update labels
            self.velocity_label.setText(f"Velocity: {velocity:.2f} m/s")
            self.force_label.setText(f"Force: {force:.2f} N")
            self.power_label.setText(f"Power: {power:.2f} W")
            
            # Calculate ROM
            if len(self.current_lift['positions']) > 1:
                start_pos = self.current_lift['positions'][0]
                current_pos = self.current_lift['positions'][-1]
                rom = abs(current_pos[1] - start_pos[1]) * self.calibration_factor
                self.rom_label.setText(f"ROM: {rom:.2f} m")
            
            # Calculate RFD (Rate of Force Development)
            if len(self.current_lift['forces']) > 1:
                force_change = self.current_lift['forces'][-1] - self.current_lift['forces'][0]
                time_change = (self.current_lift['timestamps'][-1] - 
                             self.current_lift['timestamps'][0])
                rfd = force_change / time_change if time_change > 0 else 0
                self.rfd_label.setText(f"RFD: {rfd:.2f} N/s")
            
            # Update plots
            self.update_real_time_plots()
            
        except Exception as e:
            self.logger.error(f"Metrics display error: {str(e)}")

    def update_real_time_plots(self):
        """Update real-time performance plots"""
        try:
            self.figure.clear()
            
            # Create subplots
            ax1 = self.figure.add_subplot(311)
            ax2 = self.figure.add_subplot(312)
            ax3 = self.figure.add_subplot(313)
            
            # Plot data
            times = [t - self.current_lift['timestamps'][0] 
                    for t in self.current_lift['timestamps']]
            
            ax1.plot(times, self.current_lift['velocities'], 'b-')
            ax1.set_ylabel('Velocity (m/s)')
            
            ax2.plot(times, self.current_lift['forces'], 'r-')
            ax2.set_ylabel('Force (N)')
            
            ax3.plot(times, self.current_lift['power'], 'g-')
            ax3.set_ylabel('Power (W)')
            ax3.set_xlabel('Time (s)')
            
            self.figure.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            self.logger.error(f"Plot update error: {str(e)}")

    def process_lift_data(self):
        """Process and save lift data"""
        try:
            # Calculate summary metrics
            metrics = {
                'timestamp': datetime.now(),
                'exercise_type': self.exercise_combo.currentText(),
                'weight_kg': self.weight_spin.value(),
                'peak_velocity': max(self.current_lift['velocities']),
                'mean_velocity': np.mean(self.current_lift['velocities']),
                'peak_force': max(self.current_lift['forces']),
                'mean_force': np.mean(self.current_lift['forces']),
                'peak_power': max(self.current_lift['power']),
                'mean_power': np.mean(self.current_lift['power']),
                'rom': (abs(self.current_lift['positions'][-1][1] - 
                          self.current_lift['positions'][0][1]) * 
                       self.calibration_factor)
            }
            
            # Save to database
            self.save_lift_data(metrics)
            
            # Update displays
            self.update_history_table()
            self.update_analysis()
            
            self.statusBar().showMessage("Lift data processed and saved")
            
        except Exception as e:
            self.logger.error(f"Data processing error: {str(e)}")
            self.statusBar().showMessage(f"Data processing error: {str(e)}")

    def save_lift_data(self, metrics):
        """Save lift data to database"""
        try:
            # Implement database save operation here
            pass
        except Exception as e:
            self.logger.error(f"Database error: {str(e)}")

    def update_history_table(self):
        """Update the lift history table"""
        try:
            # Implement history table update here
            pass
        except Exception as e:
            self.logger.error(f"History update error: {str(e)}")

    def update_analysis(self):
        """Update analysis charts"""
        try:
            analysis_type = self.analysis_type.currentText()
            
            self.analysis_figure.clear()
            
            if analysis_type == "Velocity Trends":
                self.plot_velocity_trends()
            elif analysis_type == "Force-Velocity Profile":
                self.plot_force_velocity_profile()
            elif analysis_type == "Power Development":
                self.plot_power_development()
            elif analysis_type == "ROM Analysis":
                self.plot_rom_analysis()
            
            self.analysis_canvas.draw()
            
        except Exception as e:
            self.logger.error(f"Analysis update error: {str(e)}")

    def save_settings(self):
        """Save application settings"""
        try:
            self.config['reference_height'] = self.reference_height.value()
            self.config['movement_threshold'] = self.movement_threshold.value()
            
            if self.camera_select.currentText() == "IP Camera":
                self.config['camera_index'] = self.ip_address.text()
            else:
                self.config['camera_index'] = 0
            
            self.save_config()
            self.statusBar().showMessage("Settings saved")
            
        except Exception as e:
            self.logger.error(f"Settings save error: {str(e)}")
            self.statusBar().showMessage(f"Settings save error: {str(e)}")

    def closeEvent(self, event):
        """Handle application closure"""
        try:
            if hasattr(self, 'cap'):
                self.cap.release()
            if hasattr(self, 'video_writer'):
                self.video_writer.release()
            event.accept()
        except Exception as e:
            self.logger.error(f"Closure error: {str(e)}")
            event.accept()

    def create_video_processing_tab(self):
        """Create tab for video file processing"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
    
        # File selection area
        file_group = QGroupBox("Video File")
        file_layout = QHBoxLayout()
        
        self.video_path_label = QLabel("No file selected")
        select_file_btn = QPushButton("Select Video")
        select_file_btn.clicked.connect(self.on_select_video)
        
        file_layout.addWidget(self.video_path_label)
        file_layout.addWidget(select_file_btn)
        file_group.setLayout(file_layout)
    
        # Processing controls
        controls_group = QGroupBox("Processing Controls")
        controls_layout = QVBoxLayout()
    
        # Exercise settings (reuse from tracking tab)
        settings = QHBoxLayout()
        settings.addWidget(QLabel("Exercise:"))
        self.video_exercise_combo = QComboBox()
        self.video_exercise_combo.addItems([
        "Squat", "Bench Press", "Deadlift", "Clean", "Snatch"
        ])
        settings.addWidget(self.video_exercise_combo)
    
        settings.addWidget(QLabel("Weight (kg):"))
        self.video_weight_spin = QDoubleSpinBox()
        self.video_weight_spin.setRange(0, 500)
        self.video_weight_spin.setValue(100)
        settings.addWidget(self.video_weight_spin)
    
        # Progress bar
        self.process_progress = QProgressBar()
        
        # Process button
        self.process_btn = QPushButton("Process Video")
        self.process_btn.clicked.connect(self.on_process_video)
        self.process_btn.setEnabled(False)
        
        controls_layout.addLayout(settings)
        controls_layout.addWidget(self.process_progress)
        controls_layout.addWidget(self.process_btn)
        controls_group.setLayout(controls_layout)
        
        # Video preview
        preview_group = QGroupBox("Video Preview")
        preview_layout = QVBoxLayout()
        self.video_preview_label = QLabel()
        self.video_preview_label.setMinimumSize(640, 480)
        preview_layout.addWidget(self.video_preview_label)
        preview_group.setLayout(preview_layout)
        
        # Add all components to main layout
        layout.addWidget(file_group)
        layout.addWidget(controls_group)
        layout.addWidget(preview_group)
        
        return widget

    def on_select_video(self):
        """Handle video file selection"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*.*)"
        )
        
        if file_path:
            self.video_path = file_path
            self.video_path_label.setText(file_path)
            self.process_btn.setEnabled(True)
            
            # Show first frame as preview
            cap = cv2.VideoCapture(file_path)
            ret, frame = cap.read()
            if ret:
                self.display_preview_frame(frame)
            cap.release()

    def display_preview_frame(self, frame):
        """Display frame in preview label"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, 
                     QImage.Format.Format_RGB888)
    
    # Scale image to fit label while maintaining aspect ratio
        scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
            self.video_preview_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio
        )
        self.video_preview_label.setPixmap(scaled_pixmap)

    def on_process_video(self):
        """Process the selected video file"""
        try:
            self.process_btn.setEnabled(False)
            self.statusBar().showMessage("Processing video...")
            
            # Create worker thread for processing
            self.process_thread = VideoProcessingThread(
            self.video_path,
            self.video_exercise_combo.currentText(),
            self.video_weight_spin.value()
        )
        
            # Connect signals
            self.process_thread.progress.connect(self.process_progress.setValue)
            self.process_thread.frame_processed.connect(self.display_preview_frame)
            self.process_thread.finished.connect(self.on_processing_complete)
            self.process_thread.error.connect(self.on_processing_error)
            
            # Start processing
            self.process_thread.start()
        
        except Exception as e:
            self.statusBar().showMessage(f"Error: {str(e)}")
            self.process_btn.setEnabled(True)

    
