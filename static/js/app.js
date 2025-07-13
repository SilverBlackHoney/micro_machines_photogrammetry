
// Global JavaScript utilities for Photogrammetry GUI

// Utility functions
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleString();
}

function showToast(message, type = 'info') {
    const toastContainer = document.getElementById('toast-container') || createToastContainer();
    
    const toast = document.createElement('div');
    toast.className = `toast align-items-center text-white bg-${type} border-0`;
    toast.setAttribute('role', 'alert');
    toast.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">
                ${message}
            </div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
        </div>
    `;
    
    toastContainer.appendChild(toast);
    
    const bsToast = new bootstrap.Toast(toast);
    bsToast.show();
    
    toast.addEventListener('hidden.bs.toast', () => {
        toast.remove();
    });
}

function createToastContainer() {
    const container = document.createElement('div');
    container.id = 'toast-container';
    container.className = 'toast-container position-fixed top-0 end-0 p-3';
    container.style.zIndex = '1055';
    document.body.appendChild(container);
    return container;
}

// API helper functions
async function apiCall(url, options = {}) {
    try {
        const response = await fetch(url, {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        });
        
        if (!response.ok) {
            throw new Error(`API call failed: ${response.status} ${response.statusText}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('API call error:', error);
        showToast('Network error. Please check your connection.', 'danger');
        throw error;
    }
}

// Progress tracking
class ProgressTracker {
    constructor(elementId) {
        this.element = document.getElementById(elementId);
        this.progressBar = this.element?.querySelector('.progress-bar');
    }
    
    update(percentage, text = '') {
        if (this.progressBar) {
            this.progressBar.style.width = `${percentage}%`;
            this.progressBar.setAttribute('aria-valuenow', percentage);
            
            if (text) {
                this.progressBar.textContent = text;
            } else {
                this.progressBar.textContent = `${percentage}%`;
            }
        }
    }
    
    show() {
        if (this.element) {
            this.element.style.display = 'block';
        }
    }
    
    hide() {
        if (this.element) {
            this.element.style.display = 'none';
        }
    }
    
    setColor(type) {
        if (this.progressBar) {
            this.progressBar.className = `progress-bar progress-bar-striped bg-${type}`;
        }
    }
}

// File upload handler
class FileUploadHandler {
    constructor(options = {}) {
        this.maxFileSize = options.maxFileSize || 10 * 1024 * 1024; // 10MB default
        this.allowedTypes = options.allowedTypes || ['image/jpeg', 'image/png', 'image/gif'];
        this.onProgress = options.onProgress || (() => {});
        this.onSuccess = options.onSuccess || (() => {});
        this.onError = options.onError || (() => {});
    }
    
    validateFile(file) {
        if (file.size > this.maxFileSize) {
            throw new Error(`File ${file.name} is too large. Maximum size is ${formatFileSize(this.maxFileSize)}`);
        }
        
        if (!this.allowedTypes.includes(file.type)) {
            throw new Error(`File ${file.name} has unsupported type. Allowed types: ${this.allowedTypes.join(', ')}`);
        }
        
        return true;
    }
    
    async uploadFiles(files, endpoint) {
        const formData = new FormData();
        
        // Validate all files first
        for (const file of files) {
            this.validateFile(file);
            formData.append('files', file);
        }
        
        try {
            const xhr = new XMLHttpRequest();
            
            xhr.upload.addEventListener('progress', (e) => {
                if (e.lengthComputable) {
                    const percentage = Math.round((e.loaded / e.total) * 100);
                    this.onProgress(percentage);
                }
            });
            
            xhr.addEventListener('load', () => {
                if (xhr.status === 200) {
                    const response = JSON.parse(xhr.responseText);
                    this.onSuccess(response);
                } else {
                    const error = JSON.parse(xhr.responseText);
                    this.onError(new Error(error.error || 'Upload failed'));
                }
            });
            
            xhr.addEventListener('error', () => {
                this.onError(new Error('Upload failed due to network error'));
            });
            
            xhr.open('POST', endpoint);
            xhr.send(formData);
            
        } catch (error) {
            this.onError(error);
        }
    }
}

// Auto-refresh handler
class AutoRefresh {
    constructor(callback, interval = 5000) {
        this.callback = callback;
        this.interval = interval;
        this.timeoutId = null;
        this.isRunning = false;
    }
    
    start() {
        if (this.isRunning) return;
        
        this.isRunning = true;
        this.scheduleNext();
    }
    
    stop() {
        if (this.timeoutId) {
            clearTimeout(this.timeoutId);
            this.timeoutId = null;
        }
        this.isRunning = false;
    }
    
    scheduleNext() {
        if (!this.isRunning) return;
        
        this.timeoutId = setTimeout(() => {
            if (this.isRunning) {
                this.callback();
                this.scheduleNext();
            }
        }, this.interval);
    }
    
    setInterval(interval) {
        this.interval = interval;
        if (this.isRunning) {
            this.stop();
            this.start();
        }
    }
}

// Initialize common functionality
document.addEventListener('DOMContentLoaded', () => {
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    const tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Initialize popovers
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    const popoverList = popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
    
    // Global error handler
    window.addEventListener('error', (e) => {
        console.error('Global error:', e.error);
        showToast('An unexpected error occurred. Please refresh the page.', 'danger');
    });
    
    // Global unhandled promise rejection handler
    window.addEventListener('unhandledrejection', (e) => {
        console.error('Unhandled promise rejection:', e.reason);
        showToast('An unexpected error occurred. Please try again.', 'danger');
    });
});

// Export utilities for use in other scripts
window.PhotogrammetryGUI = {
    formatFileSize,
    formatDate,
    showToast,
    apiCall,
    ProgressTracker,
    FileUploadHandler,
    AutoRefresh
};
