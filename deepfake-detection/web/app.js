/**
 * DeepDetect — Deepfake Detection Web App
 * ========================================
 * Handles file upload, drag-and-drop, simulated analysis, and result display.
 * When connected to a real backend, replace simulateAnalysis() with an API call.
 */

(function () {
    'use strict';

    // ── DOM References ────────────────────────────────────────────────────
    const uploadZone = document.getElementById('upload-zone');
    const fileInput = document.getElementById('file-input');
    const filePreview = document.getElementById('file-preview');
    const previewName = document.getElementById('preview-name');
    const previewSize = document.getElementById('preview-size');
    const previewRemove = document.getElementById('preview-remove');
    const btnAnalyze = document.getElementById('btn-analyze');
    const uploadCard = document.getElementById('upload-card');
    const loadingCard = document.getElementById('loading-card');
    const resultCard = document.getElementById('result-card');
    const resultImage = document.getElementById('result-image');
    const resultVideo = document.getElementById('result-video');
    const resultVerdict = document.getElementById('result-verdict');
    const confidenceValue = document.getElementById('confidence-value');
    const confidenceFill = document.getElementById('confidence-fill');
    const resultNote = document.getElementById('result-note');
    const btnAnother = document.getElementById('btn-another');
    const mobileMenuBtn = document.getElementById('mobile-menu-btn');
    const navLinks = document.getElementById('nav-links');

    let selectedFile = null;

    // ── Mobile Menu ───────────────────────────────────────────────────────
    mobileMenuBtn.addEventListener('click', function () {
        navLinks.classList.toggle('open');
    });

    // Close mobile menu when a link is clicked
    navLinks.querySelectorAll('.nav-link').forEach(function (link) {
        link.addEventListener('click', function () {
            navLinks.classList.remove('open');
        });
    });

    // ── Upload Zone — Click ───────────────────────────────────────────────
    uploadZone.addEventListener('click', function () {
        fileInput.click();
    });

    fileInput.addEventListener('change', function (e) {
        if (e.target.files && e.target.files[0]) {
            handleFile(e.target.files[0]);
        }
    });

    // ── Upload Zone — Drag & Drop ─────────────────────────────────────────
    uploadZone.addEventListener('dragover', function (e) {
        e.preventDefault();
        uploadZone.classList.add('drag-over');
    });

    uploadZone.addEventListener('dragleave', function (e) {
        e.preventDefault();
        uploadZone.classList.remove('drag-over');
    });

    uploadZone.addEventListener('drop', function (e) {
        e.preventDefault();
        uploadZone.classList.remove('drag-over');
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    // ── File Handling ─────────────────────────────────────────────────────
    function handleFile(file) {
        var validTypes = ['image/jpeg', 'image/png', 'video/mp4'];
        if (validTypes.indexOf(file.type) === -1) {
            alert('Unsupported file type. Please upload JPG, PNG, or MP4.');
            return;
        }

        selectedFile = file;
        previewName.textContent = file.name;
        previewSize.textContent = formatFileSize(file.size);
        filePreview.style.display = 'block';
        btnAnalyze.disabled = false;
    }

    function formatFileSize(bytes) {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / 1048576).toFixed(1) + ' MB';
    }

    // ── Remove File ───────────────────────────────────────────────────────
    previewRemove.addEventListener('click', function () {
        resetUpload();
    });

    function resetUpload() {
        selectedFile = null;
        fileInput.value = '';
        filePreview.style.display = 'none';
        btnAnalyze.disabled = true;
    }

    // ── Run Analysis ──────────────────────────────────────────────────────
    btnAnalyze.addEventListener('click', function () {
        if (!selectedFile) return;

        // Show loading
        uploadCard.style.display = 'none';
        resultCard.style.display = 'none';
        loadingCard.style.display = 'block';

        // Send to real backend API
        analyzeFile(selectedFile);
    });

    /**
     * Send file to the real backend API for deepfake analysis.
     * Backend: web/server.py (Flask)
     * Endpoint: POST /api/analyze (multipart/form-data)
     */
    function analyzeFile(file) {
        var formData = new FormData();
        formData.append('file', file);

        fetch('/api/analyze', {
            method: 'POST',
            body: formData,
        })
        .then(function (res) {
            if (!res.ok) {
                return res.json().then(function (data) {
                    throw new Error(data.error || 'Analysis failed');
                });
            }
            return res.json();
        })
        .then(function (data) {
            showResult({
                verdict: data.verdict,
                confidence: data.confidence,
                note: data.note,
                file: file,
            });
        })
        .catch(function (err) {
            loadingCard.style.display = 'none';
            uploadCard.style.display = 'block';
            alert('Analysis failed: ' + err.message + '\n\nMake sure the backend is running:\n  python web/server.py --no-checkpoint');
        });
    }

    // ── Show Result ───────────────────────────────────────────────────────
    function showResult(data) {
        loadingCard.style.display = 'none';

        // Set verdict
        var isFake = data.verdict === 'FAKE';
        resultVerdict.textContent = data.verdict;
        resultVerdict.className = 'result-verdict ' + (isFake ? 'fake' : 'real');

        // Set confidence
        var conf = data.confidence.toFixed(1);
        confidenceValue.textContent = conf + '%';
        confidenceFill.className = 'confidence-fill ' + (isFake ? 'fake' : 'real');

        // Animate confidence bar
        confidenceFill.style.width = '0%';
        requestAnimationFrame(function () {
            requestAnimationFrame(function () {
                confidenceFill.style.width = conf + '%';
            });
        });

        // Set note
        resultNote.textContent = data.note;

        // Set media preview
        resultImage.style.display = 'none';
        resultVideo.style.display = 'none';

        if (data.file.type.startsWith('image/')) {
            var reader = new FileReader();
            reader.onload = function (e) {
                resultImage.src = e.target.result;
                resultImage.style.display = 'block';
            };
            reader.readAsDataURL(data.file);
        } else if (data.file.type.startsWith('video/')) {
            var url = URL.createObjectURL(data.file);
            resultVideo.src = url;
            resultVideo.style.display = 'block';
            resultVideo.play();
        }

        resultCard.style.display = 'block';
    }

    // ── Analyze Another ───────────────────────────────────────────────────
    btnAnother.addEventListener('click', function () {
        resultCard.style.display = 'none';
        uploadCard.style.display = 'block';
        resetUpload();

        // Clean up video object URL
        if (resultVideo.src && resultVideo.src.startsWith('blob:')) {
            URL.revokeObjectURL(resultVideo.src);
        }
        resultVideo.src = '';
        resultImage.src = '';

        // Scroll to upload
        document.getElementById('hero').scrollIntoView({ behavior: 'smooth' });
    });

    // ── Smooth scroll for nav links ───────────────────────────────────────
    document.querySelectorAll('a[href^="#"]').forEach(function (anchor) {
        anchor.addEventListener('click', function (e) {
            var targetId = this.getAttribute('href');
            if (targetId === '#') return;
            var target = document.querySelector(targetId);
            if (target) {
                e.preventDefault();
                target.scrollIntoView({ behavior: 'smooth' });
            }
        });
    });

})();
