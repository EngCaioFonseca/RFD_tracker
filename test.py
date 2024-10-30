"""
Test script for Barbell Velocity Tracker
"""

import sys
from PyQt6.QtWidgets import QApplication
from RFD_tracker import AdvancedGUI

def main():
    # Create application
    app = QApplication(sys.argv)
    
    # Create and show main window
    window = AdvancedGUI()
    window.show()
    
    # Start application
    sys.exit(app.exec())

if __name__ == "__main__":
    main()


