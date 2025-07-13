
#!/usr/bin/env python3
"""
Startup script for the GUI application
Ensures dependencies are installed and starts the Flask app
"""

import subprocess
import sys
import os

def install_packages():
    """Install required packages"""
    packages = [
        'opencv-python>=4.5.0',
        'opencv-contrib-python>=4.5.0', 
        'numpy>=1.20.0',
        'scipy>=1.7.0',
        'open3d>=0.13.0',
        'matplotlib>=3.5.0',
        'flask',
        'pillow',
        'werkzeug'
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to install {package}: {e}")

def main():
    print("Starting Enhanced Photogrammetry GUI...")
    
    # Install dependencies
    print("Checking dependencies...")
    install_packages()
    
    # Change to src directory and start the app
    os.chdir('src')
    
    # Import and run the app
    try:
        from gui_app import app
        print("Starting Flask app on http://0.0.0.0:5000")
        app.run(host='0.0.0.0', port=5000, debug=True)
    except Exception as e:
        print(f"Failed to start GUI: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
