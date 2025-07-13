
#!/usr/bin/env python3
"""
GUI Application for Enhanced Photogrammetry Pipeline
Provides a user-friendly web interface for 3D reconstruction
"""

import os
import sys
import json
import uuid
import shutil
import threading
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image

# Add src to path
sys.path.append(os.path.dirname(__file__))

try:
    from enhanced_photogrammetry import EnhancedPhotogrammetryPipeline
    from photogrammetry_v2 import PhotogrammetryPipeline
except ImportError as e:
    print(f"Warning: Some modules could not be imported: {e}")
    # We'll handle this gracefully in the routes

app = Flask(__name__, template_folder='../templates', static_folder='../static')
app.secret_key = 'photogrammetry_gui_secret_key'

# Configuration
UPLOAD_FOLDER = 'static/uploads'
RECONSTRUCTION_FOLDER = 'static/reconstructions'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RECONSTRUCTION_FOLDER, exist_ok=True)
os.makedirs('static/css', exist_ok=True)
os.makedirs('static/js', exist_ok=True)
os.makedirs('templates', exist_ok=True)

# Global storage for reconstruction sessions
reconstruction_sessions = {}
processing_logs = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_image_info(image_path):
    """Get basic image information"""
    try:
        img = cv2.imread(image_path)
        if img is not None:
            h, w, c = img.shape
            file_size = os.path.getsize(image_path)
            return {
                'width': w,
                'height': h,
                'channels': c,
                'size_mb': round(file_size / (1024 * 1024), 2)
            }
    except:
        pass
    return None

def encode_image_to_base64(image_path):
    """Convert image to base64 for web display"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return None

class ReconstructionLogger:
    def __init__(self, session_id):
        self.session_id = session_id
        self.logs = []
        
    def log(self, level, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = {
            'timestamp': timestamp,
            'level': level,
            'message': message
        }
        self.logs.append(log_entry)
        
        # Store in global logs
        if self.session_id not in processing_logs:
            processing_logs[self.session_id] = []
        processing_logs[self.session_id].append(log_entry)

def run_reconstruction_task(session_id, images_folder, method, meshing_method):
    """Background task for running reconstruction"""
    logger = ReconstructionLogger(session_id)
    
    try:
        logger.log("INFO", f"Starting reconstruction with {method} + {meshing_method}")
        
        # Update session status
        reconstruction_sessions[session_id]['status'] = 'processing'
        reconstruction_sessions[session_id]['progress'] = 10
        
        # Initialize pipeline
        pipeline = EnhancedPhotogrammetryPipeline(images_folder)
        logger.log("INFO", "Pipeline initialized")
        
        reconstruction_sessions[session_id]['progress'] = 20
        
        # Load and analyze images
        images = pipeline.load_images()
        logger.log("INFO", f"Loaded {len(images)} images")
        
        reconstruction_sessions[session_id]['progress'] = 30
        reconstruction_sessions[session_id]['num_images'] = len(images)
        
        # Extract features
        pipeline.extract_features()
        logger.log("INFO", "Features extracted")
        
        reconstruction_sessions[session_id]['progress'] = 40
        
        # Match features
        pipeline.match_features()
        logger.log("INFO", f"Found {len(pipeline.matches)} image pairs with matches")
        
        reconstruction_sessions[session_id]['progress'] = 50
        
        # Estimate camera parameters
        pipeline.estimate_camera_intrinsics()
        pipeline.initialize_two_view_reconstruction()
        logger.log("INFO", "Initial reconstruction completed")
        
        reconstruction_sessions[session_id]['progress'] = 70
        
        # Run dense reconstruction
        output_dir = os.path.join(RECONSTRUCTION_FOLDER, session_id)
        mesh = pipeline.run_complete_dense_pipeline(
            reconstruction_method=method,
            meshing_method=meshing_method,
            output_dir=output_dir
        )
        
        reconstruction_sessions[session_id]['progress'] = 90
        logger.log("INFO", "Dense reconstruction completed")
        
        # Store results
        reconstruction_sessions[session_id]['status'] = 'completed'
        reconstruction_sessions[session_id]['progress'] = 100
        reconstruction_sessions[session_id]['output_dir'] = output_dir
        reconstruction_sessions[session_id]['mesh_available'] = mesh is not None
        
        if mesh is not None:
            logger.log("SUCCESS", f"Reconstruction successful! Generated mesh with {len(mesh.vertices)} vertices")
        else:
            logger.log("WARNING", "Reconstruction completed but no mesh was generated")
            
    except Exception as e:
        logger.log("ERROR", f"Reconstruction failed: {str(e)}")
        reconstruction_sessions[session_id]['status'] = 'failed'
        reconstruction_sessions[session_id]['error'] = str(e)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload')
def upload_page():
    return render_template('upload.html')

@app.route('/api/upload', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        return jsonify({'error': 'No files selected'}), 400
    
    # Create session
    session_id = str(uuid.uuid4())
    session_folder = os.path.join(UPLOAD_FOLDER, session_id)
    os.makedirs(session_folder, exist_ok=True)
    
    uploaded_files = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(session_folder, filename)
            file.save(file_path)
            
            # Get image info
            info = get_image_info(file_path)
            uploaded_files.append({
                'filename': filename,
                'path': file_path,
                'info': info
            })
    
    # Store session info
    reconstruction_sessions[session_id] = {
        'id': session_id,
        'uploaded_files': uploaded_files,
        'created_at': datetime.now().isoformat(),
        'status': 'uploaded',
        'progress': 0
    }
    
    return jsonify({
        'session_id': session_id,
        'uploaded_files': len(uploaded_files),
        'files': uploaded_files
    })

@app.route('/session/<session_id>')
def session_page(session_id):
    if session_id not in reconstruction_sessions:
        return "Session not found", 404
    
    session = reconstruction_sessions[session_id]
    return render_template('session.html', session=session, session_id=session_id)

@app.route('/api/session/<session_id>')
def get_session(session_id):
    if session_id not in reconstruction_sessions:
        return jsonify({'error': 'Session not found'}), 404
    
    session = reconstruction_sessions[session_id]
    
    # Add image previews
    for file_info in session.get('uploaded_files', []):
        if os.path.exists(file_info['path']):
            file_info['base64'] = encode_image_to_base64(file_info['path'])
    
    return jsonify(session)

@app.route('/api/start_reconstruction', methods=['POST'])
def start_reconstruction():
    data = request.get_json()
    session_id = data.get('session_id')
    method = data.get('method', 'patchmatch')
    meshing_method = data.get('meshing_method', 'tsdf')
    
    if session_id not in reconstruction_sessions:
        return jsonify({'error': 'Session not found'}), 404
    
    session = reconstruction_sessions[session_id]
    if session['status'] != 'uploaded':
        return jsonify({'error': 'Session not ready for reconstruction'}), 400
    
    # Start background reconstruction task
    images_folder = os.path.join(UPLOAD_FOLDER, session_id)
    thread = threading.Thread(
        target=run_reconstruction_task,
        args=(session_id, images_folder, method, meshing_method)
    )
    thread.daemon = True
    thread.start()
    
    return jsonify({'message': 'Reconstruction started'})

@app.route('/api/logs/<session_id>')
def get_logs(session_id):
    logs = processing_logs.get(session_id, [])
    return jsonify({'logs': logs})

@app.route('/viewer/<session_id>')
def viewer_page(session_id):
    if session_id not in reconstruction_sessions:
        return "Session not found", 404
    
    session = reconstruction_sessions[session_id]
    return render_template('viewer.html', session=session, session_id=session_id)

@app.route('/api/reconstruction_files/<session_id>')
def get_reconstruction_files(session_id):
    if session_id not in reconstruction_sessions:
        return jsonify({'error': 'Session not found'}), 404
    
    session = reconstruction_sessions[session_id]
    output_dir = session.get('output_dir')
    
    if not output_dir or not os.path.exists(output_dir):
        return jsonify({'files': []})
    
    files = []
    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        if os.path.isfile(file_path):
            file_size = os.path.getsize(file_path)
            files.append({
                'filename': filename,
                'size_mb': round(file_size / (1024 * 1024), 2),
                'download_url': f'/download/{session_id}/{filename}'
            })
    
    return jsonify({'files': files})

@app.route('/download/<session_id>/<filename>')
def download_file(session_id, filename):
    if session_id not in reconstruction_sessions:
        return "Session not found", 404
    
    session = reconstruction_sessions[session_id]
    output_dir = session.get('output_dir')
    
    if not output_dir:
        return "No reconstruction output available", 404
    
    file_path = os.path.join(output_dir, filename)
    if not os.path.exists(file_path):
        return "File not found", 404
    
    return send_file(file_path, as_attachment=True)

@app.route('/api/sessions')
def list_sessions():
    sessions = []
    for session_id, session in reconstruction_sessions.items():
        sessions.append({
            'id': session_id,
            'created_at': session.get('created_at'),
            'status': session.get('status'),
            'num_files': len(session.get('uploaded_files', [])),
            'progress': session.get('progress', 0)
        })
    
    # Sort by creation time (newest first)
    sessions.sort(key=lambda x: x['created_at'], reverse=True)
    return jsonify({'sessions': sessions})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
