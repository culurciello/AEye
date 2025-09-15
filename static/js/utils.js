// Global variables
let currentDate = null;
let currentHour = null;
let timelineData = {};

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    // Load theme preference
    if (localStorage.getItem('darkMode') === 'true') {
        document.body.classList.add('dark');
        updateThemeToggle();
    }

    loadAvailableDates();
    loadStats();
});

function toggleTheme() {
    document.body.classList.toggle('dark');
    const isDark = document.body.classList.contains('dark');
    localStorage.setItem('darkMode', isDark);
    updateThemeToggle();
}

function updateThemeToggle() {
    const toggle = document.querySelector('.theme-toggle');
    const isDark = document.body.classList.contains('dark');
    toggle.innerHTML = isDark ? '☀️ Light Mode' : '🌙 Dark Mode';
}

async function loadAvailableDates() {
    try {
        const response = await fetch('/api/available_dates');
        const dates = await response.json();

        const selector = document.getElementById('dateSelector');
        selector.innerHTML = '';

        if (dates.length === 0) {
            selector.innerHTML = '<option value="">No data available</option>';
            return;
        }

        dates.forEach(date => {
            const option = document.createElement('option');
            option.value = date;
            option.textContent = new Date(date).toLocaleDateString();
            selector.appendChild(option);
        });

        // Select the most recent date
        selector.value = dates[0];
        currentDate = dates[0];

        // Add change listener
        selector.addEventListener('change', (e) => {
            currentDate = e.target.value;
            loadTimelineData();
            hideEvents();
        });

        loadTimelineData();

    } catch (error) {
        console.error('Error loading dates:', error);
    }
}

async function loadStats() {
    try {
        const [motionResponse, faceResponse, objectResponse] = await Promise.all([
            fetch('/api/stats'),
            fetch('/api/face_stats'),
            fetch('/api/object_stats')
        ]);

        const motionStats = await motionResponse.json();
        const faceStats = await faceResponse.json();
        const objectStats = await objectResponse.json();

        document.getElementById('totalEvents').textContent = motionStats.total_motion_events;
        document.getElementById('totalFaces').textContent = faceStats.total_face_detections;
        document.getElementById('totalObjects').textContent = objectStats.total_object_detections;
        document.getElementById('processedEvents').textContent = motionStats.processed_events;
        document.getElementById('avgConfidence').textContent =
            (faceStats.avg_confidence * 100).toFixed(1) + '%';

        // Calculate peak hour from timeline data if available
        if (Object.keys(timelineData).length > 0) {
            const peakHour = Object.entries(timelineData)
                .sort(([,a], [,b]) => b.length - a.length)[0];
            document.getElementById('peakHour').textContent =
                peakHour ? formatHour(peakHour[0]) : 'N/A';
        }

    } catch (error) {
        console.error('Error loading stats:', error);
    }
}

async function loadTimelineData() {
    if (!currentDate) return;

    document.getElementById('timelineLoading').style.display = 'block';
    document.getElementById('timeline').style.display = 'none';
    document.getElementById('emptyState').style.display = 'none';

    try {
        const response = await fetch(`/api/hourly_activity?date=${currentDate}`);
        const data = await response.json();

        timelineData = data;
        renderTimeline();

        // Update peak hour in stats
        if (Object.keys(data).length > 0) {
            const peakHour = Object.entries(data)
                .sort(([,a], [,b]) => b.length - a.length)[0];
            document.getElementById('peakHour').textContent =
                peakHour ? formatHour(peakHour[0]) : 'N/A';
        } else {
            document.getElementById('emptyState').style.display = 'block';
        }

    } catch (error) {
        console.error('Error loading timeline data:', error);
    } finally {
        document.getElementById('timelineLoading').style.display = 'none';
    }
}

function renderTimeline() {
    const timeline = document.getElementById('timeline');
    timeline.innerHTML = '';
    timeline.style.display = 'grid';

    // Create 24 hour blocks
    for (let hour = 0; hour < 24; hour++) {
        const hourStr = hour.toString().padStart(2, '0');
        const events = timelineData[hourStr] || [];
        const eventCount = events.length;

        const hourBlock = document.createElement('div');
        hourBlock.className = 'hour-block';
        hourBlock.onclick = () => selectHour(hourStr);

        // Add activity classes
        if (eventCount > 0) {
            hourBlock.classList.add('has-activity');
            if (eventCount >= 5) {
                hourBlock.classList.add('high-activity');
            }
        }

        hourBlock.innerHTML = `
            <div class="hour-time">${formatHour(hourStr)}</div>
            ${eventCount > 0 ? `<div class="hour-count">${eventCount}</div>` : ''}
        `;

        timeline.appendChild(hourBlock);
    }
}

function selectHour(hour) {
    currentHour = hour;

    // Update active hour
    document.querySelectorAll('.hour-block').forEach((block, index) => {
        block.classList.remove('active');
        if (index === parseInt(hour)) {
            block.classList.add('active');
        }
    });

    // Load and show events for this hour
    loadHourEvents(hour);
}

async function loadHourEvents(hour) {
    const events = timelineData[hour] || [];

    if (events.length === 0) {
        hideEvents();
        return;
    }

    const container = document.getElementById('eventsContainer');
    const grid = document.getElementById('eventsGrid');
    const title = document.getElementById('eventsTitle');

    title.textContent = `Events at ${formatHour(hour)} (${events.length})`;
    container.style.display = 'block';

    grid.innerHTML = '';

    // Load detailed event data
    for (const eventId of events) {
        try {
            const response = await fetch(`/api/motion_event/${eventId}`);
            const event = await response.json();

            const eventCard = createEventCard(event);
            grid.appendChild(eventCard);
        } catch (error) {
            console.error('Error loading event:', eventId, error);
        }
    }
}

function createEventCard(event) {
    const card = document.createElement('div');
    card.className = 'event-card';
    card.onclick = () => showEventDetails(event.id);

    const startTime = new Date(event.start_time);
    const duration = parseFloat(event.duration_seconds);

    // Build object classes display
    let objectsDisplay = '';
    if (event.object_detections && event.object_detections.length > 0) {
        const objectClasses = {};
        event.object_detections.forEach(od => {
            if (!objectClasses[od.class_name]) {
                objectClasses[od.class_name] = [];
            }
            objectClasses[od.class_name].push(od);
        });

        objectsDisplay = Object.entries(objectClasses).map(([className, detections]) => {
            const firstDetection = detections[0];
            return `
                <div class="object-class-item">
                    <img src="/api/object_detection/${firstDetection.id}/image"
                         class="object-thumbnail"
                         alt="${className}"
                         onerror="this.style.display='none'">
                    <span class="object-class-label">${className} (${detections.length})</span>
                </div>
            `;
        }).join('');
    }

    card.innerHTML = `
        <div class="event-header">
            <div class="event-time">${startTime.toLocaleTimeString()}</div>
            <div class="event-duration">${duration.toFixed(1)}s</div>
        </div>

        ${objectsDisplay ? `
            <div class="object-detections-preview">
                ${objectsDisplay}
            </div>
        ` : ''}

        <div class="event-info">
            <div>
                ${event.face_count > 0 ?
                    `<span class="face-badge">${event.face_count} faces</span>` :
                    '<span style="color: var(--text-secondary);">No faces</span>'
                }
                ${event.object_count > 0 ?
                    `<span class="object-badge">${event.object_count} objects</span>` :
                    ''
                }
            </div>
            <div>
                ${event.processed ?
                    '<span class="processed-badge">Processed</span>' :
                    '<span class="unprocessed-badge">Pending</span>'
                }
            </div>
        </div>
    `;

    return card;
}

async function showEventDetails(eventId) {
    try {
        const response = await fetch(`/api/motion_event/${eventId}`);
        const event = await response.json();

        const modal = document.getElementById('eventModal');
        const title = document.getElementById('modalTitle');
        const body = document.getElementById('modalBody');

        title.textContent = `Event ${event.id} - ${new Date(event.start_time).toLocaleString()}`;

        body.innerHTML = `
            <div style="margin-bottom: 1rem;">
                <strong>Duration:</strong> ${parseFloat(event.duration_seconds).toFixed(1)} seconds<br>
                <strong>Video:</strong> ${event.video_file}<br>
                <strong>Face Count:</strong> ${event.face_count}<br>
                <strong>Object Count:</strong> ${event.object_count}<br>
                <strong>Processed:</strong> ${event.processed ? 'Yes' : 'No'}
            </div>

            ${event.object_detections && event.object_detections.length > 0 ? `
                <h4 style="margin-bottom: 0.5rem;">Object Detections (${event.object_detections.length})</h4>
                <div style="display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 1rem;">
                    ${event.object_detections.map(od => `
                        <div style="border: 1px solid var(--border-light); padding: 0.5rem; border-radius: 4px; background: var(--background);">
                            <div style="font-weight: 600; color: var(--warning-color);">${od.class_name}</div>
                            <div style="font-size: 0.75rem; color: var(--text-secondary);">
                                ${(od.confidence * 100).toFixed(1)}% confidence
                            </div>
                        </div>
                    `).join('')}
                </div>
            ` : ''}

            ${event.face_detections && event.face_detections.length > 0 ? `
                <h4 style="margin-bottom: 0.5rem;">Face Detections (${event.face_detections.length})</h4>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 0.5rem; margin-bottom: 1rem;">
                    ${event.face_detections.map(fd => `
                        <div style="border: 1px solid var(--border-light); padding: 0.5rem; border-radius: 4px; text-align: center;">
                            <img src="/api/face_detection/${fd.id}/image"
                                 style="width: 80px; height: 80px; object-fit: cover; border-radius: 4px; margin-bottom: 0.25rem;"
                                 onerror="this.style.display='none'">
                            <div style="font-size: 0.75rem; color: var(--text-secondary);">
                                ${(fd.confidence * 100).toFixed(1)}% confidence
                            </div>
                        </div>
                    `).join('')}
                </div>
            ` : ''}

            ${!event.object_detections?.length && !event.face_detections?.length ?
                '<p style="color: var(--text-secondary); margin-bottom: 1rem;">No detections for this event.</p>' : ''}

            <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid var(--border-light);">
                <video controls style="width: 100%; max-height: 300px; border-radius: 4px;">
                    <source src="/api/serve_video/${event.video_file}" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
            </div>
        `;

        modal.style.display = 'block';

    } catch (error) {
        console.error('Error loading event details:', error);
    }
}

function closeModal() {
    document.getElementById('eventModal').style.display = 'none';
}

function hideEvents() {
    document.getElementById('eventsContainer').style.display = 'none';
    currentHour = null;

    // Remove active hour
    document.querySelectorAll('.hour-block').forEach(block => {
        block.classList.remove('active');
    });
}

function formatHour(hour) {
    const h = parseInt(hour);
    if (h === 0) return '12 AM';
    if (h < 12) return `${h} AM`;
    if (h === 12) return '12 PM';
    return `${h - 12} PM`;
}

// Close modal when clicking outside
window.onclick = function(event) {
    const modal = document.getElementById('eventModal');
    if (event.target === modal) {
        closeModal();
    }
}