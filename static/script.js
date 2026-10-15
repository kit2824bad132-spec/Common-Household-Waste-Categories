/**
 * EcoSort - Household Waste Classification
 * Frontend interaction, drag & drop image upload, and API dispatch logic.
 */

document.addEventListener('DOMContentLoaded', () => {
    // -------------------------------------------------------------------------
    // 1. Mobile Navigation Toggle
    // -------------------------------------------------------------------------
    const mobileMenuBtn = document.getElementById('mobileMenuBtn');
    const navLinks = document.getElementById('navLinks');

    if (mobileMenuBtn && navLinks) {
        mobileMenuBtn.addEventListener('click', () => {
            navLinks.classList.toggle('show');
        });

        // Close mobile menu when a nav link is clicked
        navLinks.querySelectorAll('a').forEach(link => {
            link.addEventListener('click', () => {
                navLinks.classList.remove('show');
            });
        });
    }

    // -------------------------------------------------------------------------
    // 2. Active Navigation Tracking on Scroll
    // -------------------------------------------------------------------------
    const sections = document.querySelectorAll('section[id]');
    const navItems = document.querySelectorAll('.nav-link');

    window.addEventListener('scroll', () => {
        let currentSectionId = '';
        const scrollPosition = window.scrollY + 140;

        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            const sectionHeight = section.offsetHeight;
            if (scrollPosition >= sectionTop && scrollPosition < sectionTop + sectionHeight) {
                currentSectionId = section.getAttribute('id');
            }
        });

        navItems.forEach(item => {
            item.classList.remove('active');
            if (item.getAttribute('href') === `#${currentSectionId}`) {
                item.classList.add('active');
            }
        });
    });

    // -------------------------------------------------------------------------
    // 3. Image Upload & Preview Logic
    // -------------------------------------------------------------------------
    const dropzone = document.getElementById('dropzone');
    const wasteImageInput = document.getElementById('wasteImageInput');
    const dropzonePrompt = document.getElementById('dropzonePrompt');
    const previewContainer = document.getElementById('previewContainer');
    const imagePreview = document.getElementById('imagePreview');
    const changeImageBtn = document.getElementById('changeImageBtn');
    const resetBtn = document.getElementById('resetBtn');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const uploadForm = document.getElementById('uploadForm');
    const spinner = document.getElementById('spinner');

    // Result elements
    const emptyState = document.getElementById('emptyState');
    const resultDetails = document.getElementById('resultDetails');
    const statusTag = document.getElementById('statusTag');
    const resultFilename = document.getElementById('resultFilename');
    const resultPipelineStatus = document.getElementById('resultPipelineStatus');
    const resultTargetBin = document.getElementById('resultTargetBin');
    const resultNoticeText = document.getElementById('resultNoticeText');

    let selectedFile = null;

    // Trigger file chooser when dropzone is clicked (unless clicking on change image button)
    if (dropzone && wasteImageInput) {
        dropzone.addEventListener('click', (e) => {
            if (e.target !== changeImageBtn && !changeImageBtn?.contains(e.target)) {
                wasteImageInput.click();
            }
        });

        // Drag and Drop events
        ['dragenter', 'dragover'].forEach(eventName => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropzone.classList.add('dragover');
            });
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropzone.classList.remove('dragover');
            });
        });

        dropzone.addEventListener('drop', (e) => {
            const files = e.dataTransfer.files;
            if (files && files.length > 0) {
                handleFileSelection(files[0]);
            }
        });

        wasteImageInput.addEventListener('change', (e) => {
            const files = e.target.files;
            if (files && files.length > 0) {
                handleFileSelection(files[0]);
            }
        });
    }

    if (changeImageBtn) {
        changeImageBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            wasteImageInput.value = '';
            wasteImageInput.click();
        });
    }

    if (resetBtn) {
        resetBtn.addEventListener('click', resetUploadState);
    }

    function handleFileSelection(file) {
        // Validate MIME type
        const validTypes = ['image/jpeg', 'image/png', 'image/webp', 'image/jpg'];
        if (!validTypes.includes(file.type)) {
            alert('Please select a valid image format (JPEG, PNG, or WEBP).');
            return;
        }

        // Validate File Size (16MB maximum)
        const maxBytes = 16 * 1024 * 1024;
        if (file.size > maxBytes) {
            alert('File size exceeds the 16MB limit. Please upload a smaller photo.');
            return;
        }

        selectedFile = file;

        // Render preview via FileReader
        const reader = new FileReader();
        reader.onload = (event) => {
            imagePreview.src = event.target.result;
            dropzonePrompt.classList.add('hidden');
            previewContainer.classList.remove('hidden');

            // Enable action buttons
            analyzeBtn.removeAttribute('disabled');
            resetBtn.removeAttribute('disabled');

            // Reset results view to empty until analyze is clicked
            statusTag.textContent = 'Image Loaded';
            statusTag.className = 'status-tag active';
        };
        reader.readAsDataURL(file);
    }

    function resetUploadState() {
        selectedFile = null;
        if (wasteImageInput) wasteImageInput.value = '';
        if (imagePreview) imagePreview.src = '';

        dropzonePrompt.classList.remove('hidden');
        previewContainer.classList.add('hidden');

        analyzeBtn.setAttribute('disabled', 'true');
        resetBtn.setAttribute('disabled', 'true');

        emptyState.classList.remove('hidden');
        resultDetails.classList.add('hidden');

        statusTag.textContent = 'Ready to Scan';
        statusTag.className = 'status-tag awaiting';
    }

    // -------------------------------------------------------------------------
    // 4. Form Submission & API Request to /predict
    // -------------------------------------------------------------------------
    if (uploadForm) {
        uploadForm.addEventListener('submit', async (e) => {
            e.preventDefault();

            if (!selectedFile) {
                alert('Please select or drop an image file first.');
                return;
            }

            // Prepare multipart form data
            const formData = new FormData();
            formData.append('waste_image', selectedFile);

            // Update UI to loading state
            setLoading(true);

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (!response.ok || !data.success) {
                    throw new Error(data.error || 'Server error during image processing.');
                }

                // Render result details
                renderPredictionResponse(data);

            } catch (error) {
                console.error('Classification error:', error);
                alert(`Error: ${error.message}`);
                statusTag.textContent = 'Failed';
                statusTag.className = 'status-tag awaiting';
            } finally {
                setLoading(false);
            }
        });
    }

    function setLoading(isLoading) {
        const btnText = analyzeBtn.querySelector('.btn-text');
        if (isLoading) {
            analyzeBtn.setAttribute('disabled', 'true');
            resetBtn.setAttribute('disabled', 'true');
            btnText.textContent = 'Processing Pipeline...';
            spinner.classList.remove('hidden');
            statusTag.textContent = 'Verifying...';
            statusTag.className = 'status-tag active';
        } else {
            analyzeBtn.removeAttribute('disabled');
            resetBtn.removeAttribute('disabled');
            btnText.textContent = 'Analyze Waste';
            spinner.classList.add('hidden');
        }
    }

    function renderPredictionResponse(data) {
        emptyState.classList.add('hidden');
        resultDetails.classList.remove('hidden');

        const predCategoryTitle = document.getElementById('predCategoryTitle');
        const predStreamPill = document.getElementById('predStreamPill');
        const predConfidenceNumber = document.getElementById('predConfidenceNumber');
        const predConfidenceBar = document.getElementById('predConfidenceBar');
        const predBinTitle = document.getElementById('predBinTitle');
        const predGuidanceText = document.getElementById('predGuidanceText');
        const binIconCircle = document.getElementById('binIconCircle');
        const top3Bars = document.getElementById('top3Bars');
        const resultFilename = document.getElementById('resultFilename');

        if (resultFilename) {
            resultFilename.textContent = data.filename || selectedFile?.name || 'Uploaded photo';
        }

        if (data.model_connected && data.prediction) {
            const pred = data.prediction;
            statusTag.textContent = 'Classified';
            statusTag.className = 'status-tag active';

            if (predCategoryTitle) predCategoryTitle.textContent = pred.display_name;
            if (predStreamPill) predStreamPill.textContent = pred.stream;
            if (predConfidenceNumber) predConfidenceNumber.textContent = `${pred.confidence}%`;
            if (predConfidenceBar) predConfidenceBar.style.width = `${Math.min(pred.confidence, 100)}%`;
            if (predBinTitle) predBinTitle.textContent = `${pred.bin_color} Bin`;
            if (predGuidanceText) predGuidanceText.textContent = pred.guidance;

            // Update bin icon based on waste stream
            if (binIconCircle) {
                const streamLower = (pred.category || '').toLowerCase();
                if (streamLower === 'battery') binIconCircle.textContent = '⚡';
                else if (streamLower === 'biological') binIconCircle.textContent = '🍃';
                else if (streamLower.includes('glass')) binIconCircle.textContent = '🍾';
                else if (streamLower === 'cardboard' || streamLower === 'paper') binIconCircle.textContent = '📦';
                else if (streamLower === 'clothes' || streamLower === 'shoes') binIconCircle.textContent = '👕';
                else if (streamLower === 'metal') binIconCircle.textContent = '🥫';
                else if (streamLower === 'plastic') binIconCircle.textContent = '🧴';
                else binIconCircle.textContent = '🗑️';
            }

            // Top-3 probabilities distribution
            if (top3Bars && Array.isArray(pred.top_3)) {
                top3Bars.innerHTML = '';
                pred.top_3.forEach(item => {
                    const itemDiv = document.createElement('div');
                    itemDiv.className = 'top3-item';
                    itemDiv.innerHTML = `
                        <div class="top3-item-header">
                            <span>${item.display_name}</span>
                            <span>${item.confidence}%</span>
                        </div>
                        <div class="top3-track">
                            <div class="top3-fill" style="width: ${Math.min(item.confidence, 100)}%"></div>
                        </div>
                    `;
                    top3Bars.appendChild(itemDiv);
                });
            }

        } else {
            // Model Offline / Training
            statusTag.textContent = 'Model Offline';
            statusTag.className = 'status-tag pending-model';

            if (predCategoryTitle) predCategoryTitle.textContent = 'Model Pending';
            if (predStreamPill) predStreamPill.textContent = 'Awaiting Model Training';
            if (predConfidenceNumber) predConfidenceNumber.textContent = '0%';
            if (predConfidenceBar) predConfidenceBar.style.width = '0%';
            if (predBinTitle) predBinTitle.textContent = 'Awaiting Model Training';
            if (predGuidanceText) {
                predGuidanceText.textContent = data.message || 'Run "python train.py" to train and connect your classifier.';
            }
            if (top3Bars) top3Bars.innerHTML = '<p style="font-size: 0.8rem; color: #94a3b8;">Top-3 predictions will appear once model weights are connected.</p>';
        }
    }

    // Check model status at page load to update banner
    async function checkModelStatus() {
        try {
            const res = await fetch('/api/model-status');
            const status = await res.json();
            const banner = document.getElementById('modelStatusBanner');
            if (banner && status.model_loaded) {
                banner.style.backgroundColor = '#ecfdf5';
                banner.style.borderColor = '#a7f3d0';
                banner.querySelector('.status-icon-wrapper').style.color = '#059669';
                const textElem = banner.querySelector('.status-text');
                if (textElem) {
                    const accText = status.evaluation ? ` (Test Accuracy: ${(status.evaluation.accuracy * 100).toFixed(1)}%)` : '';
                    textElem.innerHTML = `<strong>AI Model Online:</strong> MobileNetV2 Active with ${status.class_count} classes connected${accText}.`;
                }
            }
        } catch (e) {
            console.log('Model status check skipped:', e);
        }
    }
    checkModelStatus();


    // -------------------------------------------------------------------------
    // 5. Contact Form Submission
    // -------------------------------------------------------------------------
    const contactForm = document.getElementById('contactForm');
    const contactAlert = document.getElementById('contactAlert');
    const contactSubmitBtn = document.getElementById('contactSubmitBtn');

    if (contactForm) {
        contactForm.addEventListener('submit', async (e) => {
            e.preventDefault();

            const formData = {
                name: document.getElementById('contactName').value,
                email: document.getElementById('contactEmail').value,
                subject: document.getElementById('contactSubject').value,
                message: document.getElementById('contactMessage').value
            };

            const originalBtnText = contactSubmitBtn.textContent;
            contactSubmitBtn.textContent = 'Sending...';
            contactSubmitBtn.disabled = true;

            try {
                const response = await fetch('/api/contact', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(formData)
                });

                const resData = await response.json();

                contactAlert.className = response.ok ? 'alert-box model-notice' : 'alert-box';
                contactAlert.style.backgroundColor = response.ok ? '#ecfdf5' : '#fef2f2';
                contactAlert.style.borderColor = response.ok ? '#a7f3d0' : '#fecaca';
                contactAlert.style.color = response.ok ? '#065f46' : '#991b1b';
                contactAlert.textContent = resData.message || resData.error;
                contactAlert.classList.remove('hidden');

                if (response.ok) {
                    contactForm.reset();
                }
            } catch (err) {
                contactAlert.className = 'alert-box';
                contactAlert.style.backgroundColor = '#fef2f2';
                contactAlert.style.color = '#991b1b';
                contactAlert.textContent = 'Failed to submit form. Please verify network connection.';
                contactAlert.classList.remove('hidden');
            } finally {
                contactSubmitBtn.textContent = originalBtnText;
                contactSubmitBtn.disabled = false;
            }
        });
    }
});
