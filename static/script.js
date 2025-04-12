// Configure marked.js
marked.setOptions({
    breaks: true,
    gfm: true,
    headerIds: false,
    mangle: false,
    sanitize: false,
    silent: true
});

// Helper functions for performance optimization
function debounce(func, wait) {
  let timeout;
  return function(...args) {
    clearTimeout(timeout);
    timeout = setTimeout(() => func.apply(this, args), wait);
  };
}

function throttle(func, limit) {
  let lastFunc;
  let lastRan;
  return function() {
    const context = this;
    const args = arguments;
    if (!lastRan) {
      func.apply(context, args);
      lastRan = Date.now();
    } else {
      clearTimeout(lastFunc);
      lastFunc = setTimeout(function() {
        if ((Date.now() - lastRan) >= limit) {
          func.apply(context, args);
          lastRan = Date.now();
        }
      }, limit - (Date.now() - lastRan));
    }
  };
}

// DOM Elements
const sidebar = document.getElementById('sidebar');
const sidebarToggle = document.getElementById('sidebar-toggle');
const sidebarTrigger = document.getElementById('sidebar-trigger');
const sidebarIndicator = document.getElementById('sidebar-indicator');
const mainContainer = document.getElementById('main-container');
const messagesContainer = document.getElementById('messages-container');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const micBtn = document.getElementById('mic-btn');
const imageBtn = document.getElementById('image-btn');
const clearBtn = document.getElementById('clear-btn');
const imageUpload = document.getElementById('image-upload');
const imagePreview = document.getElementById('image-preview');
const modelSelect = document.getElementById('model-select');
const languageSelect = document.getElementById('language-select');
const voiceSelect = document.getElementById('voice-select');
const ttsToggle = document.getElementById('tts-toggle');
const themeToggle = document.getElementById('theme-toggle');
const toggleContrast = document.getElementById('toggle-contrast');
const increaseFontBtn = document.getElementById('increase-font');
const decreaseFontBtn = document.getElementById('decrease-font');
const srAnnouncer = document.getElementById('sr-announcer');
const chatTitle = document.getElementById('chat-title');
const easterEggTrigger = document.getElementById('easter-egg-trigger');

// Global variables
let ws;
let mediaRecorder;
let audioChunks = [];
let isRecording = false;
let fontSizeLevel = 100;
let currentStreamingMessage = null;
let pendingImage = null;
let activeTtsPlayer = null;
let accumulatedText = '';
let currentTheme = 'light';
let konamiSequence = [];
let konamiCode = ['ArrowUp', 'ArrowUp', 'ArrowDown', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'ArrowLeft', 'ArrowRight', 'b', 'a'];
let konamiActivated = false;
let sidebarExpanded = false;
let titleSet = false;
let statusMessageId = 0;
let messageRenderQueue = [];
let isProcessingQueue = false;
const markdownCache = new Map();
let currentProcessingMessage = null; // Track the processing message
let lastUserMessage = null; // Store the last user message for proper ordering
let ariaStatus, ariaAlert; // ARIA live regions for accessibility

// Setup IntersectionObserver for lazy loading images
const imageObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      const img = entry.target;
      if (img.dataset.src) {
        img.src = img.dataset.src;
        img.removeAttribute('data-src');
        imageObserver.unobserve(img);
      }
    }
  });
}, { rootMargin: '100px' });

// Voice map to populate voice dropdown based on language
const voiceMap = {
    'a': [ // American English
        {id: 'af_heart', name: 'Heart', gender: '👩', grade: 'A', traits: '❤️'},
        {id: 'af_bella', name: 'Bella', gender: '👩', grade: 'A-', traits: '🔥'},
        {id: 'af_nicole', name: 'Nicole', gender: '👩', grade: 'B-', traits: '🎧'},
        {id: 'af_aoede', name: 'Aoede', gender: '👩', grade: 'C+', traits: ''},
        {id: 'af_kore', name: 'Kore', gender: '👩', grade: 'C+', traits: ''},
        {id: 'af_sarah', name: 'Sarah', gender: '👩', grade: 'C+', traits: ''},
        {id: 'am_fenrir', name: 'Fenrir', gender: '👨', grade: 'C+', traits: ''},
        {id: 'am_michael', name: 'Michael', gender: '👨', grade: 'C+', traits: ''},
        {id: 'am_puck', name: 'Puck', gender: '👨', grade: 'C+', traits: ''}
    ],
    'b': [ // British English
        {id: 'bf_emma', name: 'Emma', gender: '👩', grade: 'B-', traits: ''},
        {id: 'bf_isabella', name: 'Isabella', gender: '👩', grade: 'C', traits: ''},
        {id: 'bm_fable', name: 'Fable', gender: '👨', grade: 'C', traits: ''},
        {id: 'bm_george', name: 'George', gender: '👨', grade: 'C', traits: ''}
    ],
    'j': [ // Japanese
        {id: 'jf_alpha', name: 'Alpha', gender: '👩', grade: 'C+', traits: ''},
        {id: 'jf_gongitsune', name: 'Gongitsune', gender: '👩', grade: 'C', traits: ''},
        {id: 'jm_kumo', name: 'Kumo', gender: '👨', grade: 'C-', traits: ''}
    ],
    'z': [ // Mandarin Chinese
        {id: 'zf_xiaobei', name: 'Xiaobei', gender: '👩', grade: 'D', traits: ''},
        {id: 'zf_xiaoni', name: 'Xiaoni', gender: '👩', grade: 'D', traits: ''},
        {id: 'zm_yunjian', name: 'Yunjian', gender: '👨', grade: 'D', traits: ''}
    ],
    'e': [ // Spanish
        {id: 'ef_dora', name: 'Dora', gender: '👩', grade: 'C', traits: ''},
        {id: 'em_alex', name: 'Alex', gender: '👨', grade: 'C', traits: ''}
    ],
    'f': [ // French
        {id: 'ff_siwis', name: 'Siwis', gender: '👩', grade: 'B-', traits: ''}
    ],
    'h': [ // Hindi
        {id: 'hf_alpha', name: 'Alpha', gender: '👩', grade: 'C', traits: ''},
        {id: 'hf_beta', name: 'Beta', gender: '👩', grade: 'C', traits: ''}
    ],
    'i': [ // Italian
        {id: 'if_sara', name: 'Sara', gender: '👩', grade: 'C', traits: ''},
        {id: 'im_nicola', name: 'Nicola', gender: '👨', grade: 'C', traits: ''}
    ],
    'p': [ // Brazilian Portuguese
        {id: 'pf_dora', name: 'Dora', gender: '👩', grade: 'C', traits: ''},
        {id: 'pm_alex', name: 'Alex', gender: '👨', grade: 'C', traits: ''}
    ]
};

// Focus management helper for accessibility
function setFocusAfterOperation(element, operation) {
    const activeElement = document.activeElement;
    const wasKeyboard = activeElement && activeElement.tagName !== 'BODY';

    operation();

    // Only restore focus if user was navigating by keyboard
    if (wasKeyboard && element) {
        setTimeout(() => element.focus(), 100);
    }
}

// Better error handling for screen readers
function handleAndAnnounceError(message, technical = null) {
    // Log technical details for debugging
    if (technical) console.error(technical);

    // Clear any processing message
    if (currentProcessingMessage) {
        const contentDiv = currentProcessingMessage.querySelector('.message-content');
        contentDiv.innerHTML = `
            <div class="error-message">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="12" cy="12" r="10"></circle>
                    <line x1="12" y1="8" x2="12" y2="12"></line>
                    <line x1="12" y1="16" x2="12.01" y2="16"></line>
                </svg>
                <p>${message}</p>
            </div>
        `;
        currentProcessingMessage = null;
    } else {
        // Add message as error
        addMessage(`Error: ${message}`, 'assistant', null, true);
    }

    // Announce to screen readers
    announce(`Error: ${message}`, true);

    // Optional: Add haptic feedback if supported
    if (navigator.vibrate) navigator.vibrate(200);
}

// Use requestAnimationFrame for smooth scrolling
function smoothScrollToBottom(element) {
  const scrollHeight = element.scrollHeight;
  const currentPosition = element.scrollTop + element.clientHeight;
  const diff = scrollHeight - currentPosition;

  if (diff <= 1) return; // Already at bottom

  // Use smaller steps for smoother scrolling
  const step = Math.max(1, Math.floor(diff / 5));

  element.scrollTop += step;

  if (element.scrollTop + element.clientHeight < scrollHeight) {
    requestAnimationFrame(() => smoothScrollToBottom(element));
  }
}

// Queue message updates to improve performance
function queueMessageUpdate(messageData) {
  messageRenderQueue.push(messageData);
  if (!isProcessingQueue) {
    requestAnimationFrame(processMessageQueue);
  }
}

function processMessageQueue() {
  isProcessingQueue = true;

  // Process up to 5 messages at once to avoid blocking the main thread
  const batch = messageRenderQueue.splice(0, 5);

  batch.forEach(data => {
    if (data.type === 'user') {
      addMessage(data.text, 'user', data.imageSrc);
    } else if (data.type === 'text-chunk') {
      handleTextStreamChunk(data.chunk);
    }
  });

  if (messageRenderQueue.length > 0) {
    requestAnimationFrame(processMessageQueue);
  } else {
    isProcessingQueue = false;
  }
}

// Toggle sidebar (now hover-based)
function showSidebar() {
    sidebar.classList.add('sidebar-expanded');
    sidebarExpanded = true;
}

function hideSidebar() {
    sidebar.classList.remove('sidebar-expanded');
    sidebarExpanded = false;
}

function toggleSidebar() {
    if (sidebarExpanded) {
        hideSidebar();
    } else {
        showSidebar();
    }
}

// Create a processing message that will be replaced by streaming text
function createProcessingMessage(message) {
    // Create message if it doesn't exist
    if (!currentProcessingMessage) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message assistant-message';

        // Add avatar
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.textContent = 'M';
        messageDiv.appendChild(avatar);

        // Create message content container
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        messageDiv.appendChild(messageContent);

        // Create the processing container with loading spinner
        const processingDiv = document.createElement('div');
        processingDiv.className = 'processing-container';

        // Add loading spinner and message
        processingDiv.innerHTML = `
            <span class="loading"></span>
            <span class="processing-text">${message}</span>
        `;

        messageContent.appendChild(processingDiv);

        // Append to messages container at the appropriate position
        if (lastUserMessage) {
            // Insert after the last user message if it exists
            lastUserMessage.insertAdjacentElement('afterend', messageDiv);
        } else {
            // Otherwise just append to the end
            messagesContainer.appendChild(messageDiv);
        }

        // Set as current processing message
        currentProcessingMessage = messageDiv;

        // Scroll to show the message
        smoothScrollToBottom(messagesContainer);
    } else {
        // Update existing message
        const processingText = currentProcessingMessage.querySelector('.processing-text');
        if (processingText) {
            processingText.textContent = message;
        }
    }

    return currentProcessingMessage;
}

// Function to remove uploaded image
function removeUploadedImage() {
    const container = document.getElementById('image-preview-container');
    container.style.display = 'none';

    const imagePreview = document.getElementById('image-preview');
    imagePreview.src = '';
    imagePreview.alt = '';

    // Announce to screen readers
    announce('Image removed', false);
}

// Add image remove button
function addImageRemoveButton() {
    // Remove any existing button first
    const existingButton = document.getElementById('remove-image-btn');
    if (existingButton) existingButton.remove();

    // Create the remove button
    const removeBtn = document.createElement('button');
    removeBtn.id = 'remove-image-btn';
    removeBtn.className = 'image-remove-btn';
    removeBtn.innerHTML = `
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <line x1="18" y1="6" x2="6" y2="18"></line>
            <line x1="6" y1="6" x2="18" y2="18"></line>
        </svg>
    `;
    removeBtn.setAttribute('aria-label', 'Remove image');
    removeBtn.addEventListener('click', removeUploadedImage);

    // Add to container instead of the image itself
    const container = document.getElementById('image-preview-container');
    container.appendChild(removeBtn);
}

// Optimized Audio player class - with improved integration
class AudioPlayer {
    constructor(audioBase64, duration, container) {
        this.audioBase64 = audioBase64;
        this.duration = duration;
        this.container = container;
        this.audio = new Audio(`data:audio/wav;base64,${audioBase64}`);
        this.isPlaying = false;
        this.isMuted = false;
        this.progressInterval = null;

        this.createPlayerUI();
        this.setupEventListeners();

        // Autoplay
        this.play();
    }

    createPlayerUI() {
        // Create player container
        this.playerElement = document.createElement('div');
        this.playerElement.className = 'audio-player';

        // Create controls
        this.playerElement.innerHTML = `
            <div class="audio-controls">
                <button class="player-button play-pause" aria-label="Play">▶</button>
                <button class="player-button rewind" aria-label="Rewind 10 seconds">⟲</button>
                <button class="player-button forward" aria-label="Forward 10 seconds">⟳</button>
                <div class="player-progress-container">
                    <div class="player-progress-bar"></div>
                </div>
                <div class="time-display">0:00 / ${this.formatTime(this.duration)}</div>
            </div>
        `;

        // Add to container
        this.container.appendChild(this.playerElement);

        // Get elements
        this.playPauseBtn = this.playerElement.querySelector('.play-pause');
        this.rewindBtn = this.playerElement.querySelector('.rewind');
        this.forwardBtn = this.playerElement.querySelector('.forward');
        this.progressContainer = this.playerElement.querySelector('.player-progress-container');
        this.progressBar = this.playerElement.querySelector('.player-progress-bar');
        this.timeDisplay = this.playerElement.querySelector('.time-display');
    }

    setupEventListeners() {
        // Play/Pause button
        this.playPauseBtn.addEventListener('click', () => {
            if (this.isPlaying) {
                this.pause();
            } else {
                this.play();
            }
        });

        // Rewind button
        this.rewindBtn.addEventListener('click', () => {
            this.audio.currentTime = Math.max(0, this.audio.currentTime - 10);
            this.updateProgress();
        });

        // Forward button
        this.forwardBtn.addEventListener('click', () => {
            this.audio.currentTime = Math.min(this.duration, this.audio.currentTime + 10);
            this.updateProgress();
        });

        // Progress bar click - use a throttled version for better performance
        this.progressContainer.addEventListener('click', this.handleProgressClick.bind(this));

        // Audio ended event
        this.audio.addEventListener('ended', () => {
            this.isPlaying = false;
            this.playPauseBtn.innerHTML = '▶';
            this.playPauseBtn.setAttribute('aria-label', 'Play');
            clearInterval(this.progressInterval);
        });

        // Throttle progress updates for better performance
        this.audio.addEventListener('timeupdate', throttle(() => {
            this.updateProgress();
        }, 100));
    }

    handleProgressClick(e) {
        e.preventDefault();
        const rect = this.progressContainer.getBoundingClientRect();
        const pos = (e.clientX - rect.left) / rect.width;
        this.audio.currentTime = pos * this.duration;
        this.updateProgress();
    }

    play() {
        // Stop any other playing audio
        if (activeTtsPlayer && activeTtsPlayer !== this) {
            activeTtsPlayer.pause();
        }

        this.audio.play();
        this.isPlaying = true;
        this.playPauseBtn.innerHTML = '⏸';
        this.playPauseBtn.setAttribute('aria-label', 'Pause');

        // Update progress bar every 100ms
        clearInterval(this.progressInterval);
        this.progressInterval = setInterval(() => {
            this.updateProgress();
        }, 100);

        // Set this as the active player
        activeTtsPlayer = this;
    }

    pause() {
        this.audio.pause();
        this.isPlaying = false;
        this.playPauseBtn.innerHTML = '▶';
        this.playPauseBtn.setAttribute('aria-label', 'Play');
        clearInterval(this.progressInterval);
    }

    updateProgress() {
        const currentTime = this.audio.currentTime;
        const percent = (currentTime / this.duration) * 100;

        // Use requestAnimationFrame to avoid layout thrashing
        requestAnimationFrame(() => {
            this.progressBar.style.width = `${percent}%`;
            this.timeDisplay.textContent = `${this.formatTime(currentTime)} / ${this.formatTime(this.duration)}`;
        });
    }

    formatTime(seconds) {
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = Math.floor(seconds % 60);
        return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
    }

    destroy() {
        clearInterval(this.progressInterval);
        this.audio.pause();
        this.audio = null;
        if (this.playerElement.parentNode) {
            this.playerElement.parentNode.removeChild(this.playerElement);
        }
        if (activeTtsPlayer === this) {
            activeTtsPlayer = null;
        }
    }
}

// Create sparkle effect
function createSparkle(x, y) {
    const sparkle = document.createElement('div');
    sparkle.className = 'logo-sparkle';
    sparkle.style.left = `${x}px`;
    sparkle.style.top = `${y}px`;
    sparkle.style.backgroundColor = `hsl(${Math.random() * 360}, 100%, 70%)`;
    document.body.appendChild(sparkle);

    // Animate the sparkle
    setTimeout(() => {
        sparkle.style.opacity = '1';
        sparkle.style.transform = 'scale(1) rotate(180deg)';
        setTimeout(() => {
            sparkle.style.opacity = '0';
            sparkle.style.transform = 'scale(0) rotate(360deg)';
            setTimeout(() => {
                sparkle.remove();
            }, 300);
        }, 300);
    }, 10);
}

// Simplified chat title function
function generateChatTitle() {
    if (titleSet) return;
    chatTitle.textContent = "Malva Chat";
    titleSet = true;
}

// Toggle theme (light/dark/high-contrast)
function toggleTheme() {
    const body = document.body;
    const themeButton = themeToggle;

    if (currentTheme === 'light') {
        body.setAttribute('data-theme', 'dark');
        currentTheme = 'dark';
        themeButton.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>
            </svg>
        `;
        announce('Dark mode enabled');
    } else {
        body.setAttribute('data-theme', 'light');
        currentTheme = 'light';
        themeButton.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="5"></circle>
                <line x1="12" y1="1" x2="12" y2="3"></line>
                <line x1="12" y1="21" x2="12" y2="23"></line>
                <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line>
                <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line>
                <line x1="1" y1="12" x2="3" y2="12"></line>
                <line x1="21" y1="12" x2="23" y2="12"></line>
                <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line>
                <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>
            </svg>
        `;
        announce('Light mode enabled');
    }
}

// Toggle high contrast mode
function toggleHighContrast() {
    const body = document.body;
    if (body.getAttribute('data-theme') === 'high-contrast') {
        body.setAttribute('data-theme', currentTheme);
        announce('High contrast mode disabled');
    } else {
        body.setAttribute('data-theme', 'high-contrast');
        announce('High contrast mode enabled');
    }
}

// Update voice dropdown based on selected language
function updateVoiceDropdown(langCode) {
    if (!voiceMap[langCode]) {
        console.error(`No voices found for language code: ${langCode}`);
        return;
    }

    // Clear existing options
    voiceSelect.innerHTML = '';

    // Add voice options
    voiceMap[langCode].forEach(voice => {
        const option = document.createElement('option');
        option.value = voice.id;

        // Create display text with gender emoji, name, and grade
        let displayText = `${voice.gender} ${voice.name}`;

        if (voice.grade) {
            displayText += ` (${voice.grade})`;
        }

        if (voice.traits) {
            displayText += ` ${voice.traits}`;
        }

        option.textContent = displayText;
        voiceSelect.appendChild(option);
    });

    // Select first voice by default or af_heart for American English
    if (voiceMap[langCode].length > 0) {
        if (langCode === 'a') {
            // Set to af_heart for American English
            voiceSelect.value = 'af_heart';
        } else {
            // Otherwise use first voice
            voiceSelect.value = voiceMap[langCode][0].id;
        }
    }
}

// Process konami code
function processKonamiCode(key) {
    konamiSequence.push(key);

    // Keep only the last 10 keys
    if (konamiSequence.length > 10) {
        konamiSequence.shift();
    }

    // Check if the sequence matches
    const isKonami = konamiSequence.join(',') === konamiCode.join(',');

    if (isKonami && !konamiActivated) {
        konamiActivated = true;

        // Apply rainbow effect to messages
        document.querySelectorAll('.message').forEach(msg => {
            msg.classList.add('konami-active');
        });

        // Add a special message
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message assistant-message konami-active';
        messageDiv.innerHTML = `
            <div class="message-avatar">🎮</div>
            <div class="message-content">
                <div class="markdown">
                    <p>🎮 KONAMI CODE ACTIVATED! 🎮</p>
                    <p>You found a secret! How about some extra fun features?</p>
                </div>
            </div>
        `;
        messagesContainer.appendChild(messageDiv);
        smoothScrollToBottom(messagesContainer);

        // Reset after 10 seconds
        setTimeout(() => {
            document.querySelectorAll('.message').forEach(msg => {
                msg.classList.remove('konami-active');
            });
            konamiActivated = false;
        }, 10000);
    }
}

// Initialize WebSocket connection
function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;

    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        console.log('WebSocket connected');
        announce('Connected to assistant');
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleWebSocketMessage(data);
    };

    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        announce('Connection error. Please try again.', true);
    };

    ws.onclose = () => {
        console.log('WebSocket closed');
        announce('Connection closed. Reconnecting...');

        // Try to reconnect after a delay
        setTimeout(connectWebSocket, 3000);
    };
}

// Handle WebSocket messages
function handleWebSocketMessage(data) {
    console.log('Received message:', data);

    switch(data.type) {
        case 'status':
            createProcessingMessage(data.content);
            break;

        case 'status_clear':
            // No need to remove - will be replaced by streaming text
            break;

        case 'text_stream':
            // Start streaming text - which will replace the processing message
            handleTextStreamChunk(data.content);
            break;

        case 'text_stream_end':
            finishTextStream();
            break;

        case 'audio_full':
            // Audio is now added directly to the existing message
            addAudioToCurrentMessage(data.data, data.duration, data.tts_generation_time);
            break;

        case 'metrics':
            addMetricsToLastMessage(data.message_generation_time);
            break;

        case 'recognition':
            // Create editable text input with the transcription
            const recognizedText = data.content;
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message user-message edit-mode';
            messageDiv.innerHTML = `
                <div class="message-avatar">Y</div>
                <div class="message-content">
                    <textarea class="edit-transcript">${recognizedText}</textarea>
                    <div class="transcript-actions">
                        <button class="confirm-transcript">Send</button>
                        <button class="cancel-transcript">Cancel</button>
                    </div>
                </div>
            `;
            messagesContainer.appendChild(messageDiv);

            // Set up listeners
            const textarea = messageDiv.querySelector('.edit-transcript');
            const confirmBtn = messageDiv.querySelector('.confirm-transcript');
            const cancelBtn = messageDiv.querySelector('.cancel-transcript');

            // Auto-focus and select all text
            textarea.focus();
            textarea.select();

            // Confirm button sends to LLM
            confirmBtn.addEventListener('click', () => {
                const editedText = textarea.value.trim();
                if (!editedText) return;

                // Replace editable message with final version
                lastUserMessage = addMessage(editedText, 'user');
                messageDiv.remove();

                // Create processing message and send
                createProcessingMessage('Processing your message...');
                ws.send(JSON.stringify({
                    type: 'text',
                    content: editedText,
                    model: modelSelect.value,
                    lang_code: languageSelect.value,
                    voice: voiceSelect.value,
                    tts_enabled: ttsToggle.checked
                }));
            });

            // Cancel returns to recording
            cancelBtn.addEventListener('click', () => {
                messageDiv.remove();
            });

            announce('Speech recognized. Edit or send message.');
            break;

        case 'image_info':
            storeImageForNextMessage(data.image_src);
            break;

        case 'error':
            // Use improved error handling
            handleAndAnnounceError(data.content);
            break;

        case 'tts_status':
            // We handle this differently now - update the metrics in the current message
            updateTtsStatus(data.content, data.progress);
            break;

        case 'tts_status_clear':
            // Clear TTS status - no need to do anything, UI will update
            break;
    }
}

// Update TTS Status within the current message
function updateTtsStatus(content, progress) {
    if (!currentStreamingMessage && !currentProcessingMessage) return;

    const targetMessage = currentStreamingMessage || currentProcessingMessage;

    // Find or create metrics div
    let metrics = targetMessage.querySelector('.metrics');
    if (!metrics) {
        metrics = document.createElement('div');
        metrics.className = 'metrics';
        targetMessage.querySelector('.message-content').appendChild(metrics);
    }

    // Update with TTS status
    metrics.innerHTML = `${content} (${progress}%)`;

    // Add progress bar if not exists
    let progressContainer = targetMessage.querySelector('.tts-progress-container');
    if (!progressContainer && progress > 0 && progress < 100) {
        progressContainer = document.createElement('div');
        progressContainer.className = 'progress-container tts-progress-container';
        progressContainer.innerHTML = '<div class="progress-bar"></div>';
        metrics.parentNode.insertBefore(progressContainer, metrics);
    }

    // Update progress bar
    if (progressContainer) {
        const progressBar = progressContainer.querySelector('.progress-bar');
        progressBar.style.width = `${progress}%`;

        // Remove when complete
        if (progress >= 100) {
            setTimeout(() => {
                if (progressContainer.parentNode) {
                    progressContainer.parentNode.removeChild(progressContainer);
                }
            }, 500);
        }
    }
}

// Add audio directly to the current message
function addAudioToCurrentMessage(audioBase64, duration, ttsTime) {
    if (!currentStreamingMessage) return;

    // Get the message content div
    const contentDiv = currentStreamingMessage.querySelector('.message-content');
    if (!contentDiv) return;

    // Create audio player directly in this message
    new AudioPlayer(audioBase64, duration, contentDiv);

    // Update metrics
    let metrics = currentStreamingMessage.querySelector('.metrics');
    if (metrics) {
        metrics.innerHTML = `Response generated in ${metrics.dataset.responseTime || '?'}s | Speech: ${ttsTime}s`;
    }
}

// Store image to be added to next message
function storeImageForNextMessage(imageSrc) {
    pendingImage = imageSrc;
}

// Optimized markdown formatting with caching
function formatMarkdownOptimized(text) {
    // Check cache first
    if (markdownCache.has(text)) {
        return markdownCache.get(text);
    }

    // Use standard formatter
    const html = formatMarkdown(text);

    // Cache the result for future use
    if (text.length > 100) { // Only cache if worth it
        markdownCache.set(text, html);

        // Limit cache size
        if (markdownCache.size > 50) {
            const firstKey = markdownCache.keys().next().value;
            markdownCache.delete(firstKey);
        }
    }

    return html;
}

// Format markdown with code blocks enhanced
function formatMarkdown(text) {
    // Process the markdown with marked.js
    let html = marked.parse(text);

    // Add copy button to code blocks
    html = html.replace(/<pre><code class="language-([^"]+)">/g,
        '<div class="code-block-header"><span>$1</span>' +
        '<button class="code-copy-btn" onclick="copyCode(this)">Copy</button></div>' +
        '<pre class="with-header"><code class="language-$1">');

    return html;
}

// Handle text stream chunks (typing effect) - optimized version
function handleTextStreamChunk(chunk) {
    // If we have a processing message, convert it to streaming message
    if (currentProcessingMessage && !currentStreamingMessage) {
        // Keep the same message div but replace its content
        currentStreamingMessage = currentProcessingMessage;
        const contentDiv = currentStreamingMessage.querySelector('.message-content');

        // Create markdown container
        const markdownDiv = document.createElement('div');
        markdownDiv.className = 'markdown';
        contentDiv.innerHTML = ''; // Clear the processing indicator
        contentDiv.appendChild(markdownDiv);

        // Reset accumulated text
        accumulatedText = '';

        // No need to append to messages container since we're reusing the div
        currentProcessingMessage = null; // Clear reference
    }

    // If still no streaming message, create a new one
    if (!currentStreamingMessage) {
        // Create a new message for the streaming response
        currentStreamingMessage = document.createElement('div');
        currentStreamingMessage.className = 'message assistant-message';

        // Use document fragment for better performance
        const fragment = document.createDocumentFragment();

        // Add avatar
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.textContent = 'M'; // Changed from 'W' to 'M' for Malva
        fragment.appendChild(avatar);

        // Create message content container
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        fragment.appendChild(messageContent);

        // Create markdown container
        const markdownDiv = document.createElement('div');
        markdownDiv.className = 'markdown';
        messageContent.appendChild(markdownDiv);

        // Append fragment to message
        currentStreamingMessage.appendChild(fragment);

        // Reset accumulated text
        accumulatedText = '';

        // Add after the last user message if it exists
        if (lastUserMessage) {
            lastUserMessage.insertAdjacentElement('afterend', currentStreamingMessage);
        } else {
            // Otherwise append to the end
            messagesContainer.appendChild(currentStreamingMessage);
        }

        // Scroll to show the new message
        smoothScrollToBottom(messagesContainer);
    }

    // Add chunk to accumulated text
    accumulatedText += chunk;

    // Process text at most every 100ms for longer messages
    if (accumulatedText.length > 500) {
        debounceFormatMarkdown();
    } else {
        // Get the markdown div
        const markdownDiv = currentStreamingMessage.querySelector('.markdown');

        // Format the accumulated text with marked.js
        markdownDiv.innerHTML = formatMarkdownOptimized(accumulatedText);

        // Add cursor (will be appended to the end of the formatted content)
        const cursor = document.createElement('span');
        cursor.className = 'typing-cursor';
        markdownDiv.appendChild(cursor);

        // Auto-scroll efficiently
        requestAnimationFrame(() => {
            smoothScrollToBottom(messagesContainer);
        });
    }
}

// Debounced version for formatting long markdown
const debounceFormatMarkdown = debounce(() => {
    if (!currentStreamingMessage) return;

    const markdownDiv = currentStreamingMessage.querySelector('.markdown');
    if (!markdownDiv) return;

    // Format the accumulated text with marked.js
    markdownDiv.innerHTML = formatMarkdownOptimized(accumulatedText);

    // Add cursor
    const cursor = document.createElement('span');
    cursor.className = 'typing-cursor';
    markdownDiv.appendChild(cursor);

    // Auto-scroll efficiently
    requestAnimationFrame(() => {
        smoothScrollToBottom(messagesContainer);
    });
}, 100);

// Copy code function for code blocks
window.copyCode = function(button) {
    const codeBlock = button.parentElement.nextElementSibling.querySelector('code');
    const code = codeBlock.innerText;

    navigator.clipboard.writeText(code).then(() => {
        // Change button text temporarily
        const originalText = button.innerText;
        button.innerText = 'Copied!';
        setTimeout(() => {
            button.innerText = originalText;
        }, 2000);
    }).catch(err => {
        console.error('Failed to copy code: ', err);
    });
};

// Finish the text stream
function finishTextStream() {
    if (currentStreamingMessage) {
        // Get the markdown div
        const markdownDiv = currentStreamingMessage.querySelector('.markdown');

        // Format the final text with marked.js
        markdownDiv.innerHTML = formatMarkdownOptimized(accumulatedText);

        // Add image if pending
        if (pendingImage) {
            const img = document.createElement('img');
            img.dataset.src = pendingImage;
            img.alt = "Uploaded image";
            img.className = "message-image";

            // Use a placeholder initially
            img.src = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1 1'%3E%3C/svg%3E";

            markdownDiv.appendChild(img);

            // Observe the image for lazy loading
            imageObserver.observe(img);

            pendingImage = null;
        }

        // Don't reset currentStreamingMessage yet - we'll need it for the audio

        // Auto-scroll
        requestAnimationFrame(() => {
            smoothScrollToBottom(messagesContainer);
        });
    }
}

// Show alt text dialog for image accessibility
function showAltTextDialog(filename) {
    const dialog = document.createElement('div');
    dialog.className = 'alt-text-dialog';
    dialog.innerHTML = `
        <div class="alt-text-content">
            <h3>Describe this image</h3>
            <p>Adding a description helps visually impaired users understand your image.</p>
            <textarea id="alt-text-input" placeholder="Example: A red flower in a garden"
                aria-label="Image description for screen readers"></textarea>
            <div class="alt-text-actions">
                <button id="skip-alt-text">Skip</button>
                <button id="confirm-alt-text">Add Description</button>
            </div>
        </div>
    `;
    document.body.appendChild(dialog);

    // Set up event listeners
    const input = document.getElementById('alt-text-input');
    input.focus();

    document.getElementById('confirm-alt-text').addEventListener('click', () => {
        const imagePreview = document.getElementById('image-preview');
        imagePreview.alt = input.value || filename;
        dialog.remove();
        announce(`Image uploaded with description: ${imagePreview.alt}`);
    });

    document.getElementById('skip-alt-text').addEventListener('click', () => {
        const imagePreview = document.getElementById('image-preview');
        imagePreview.alt = filename;
        dialog.remove();
        announce(`Image uploaded: ${filename}`);
    });
}

// Add message to chat - optimized version
function addMessage(text, sender, imageSrc = null, isError = false) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;

    if (isError) {
        messageDiv.classList.add('error-message');
    }

    // Add avatar
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = sender === 'user' ? 'Y' : 'M'; // Changed from 'W' to 'M' for Malva
    messageDiv.appendChild(avatar);

    // Create message content container
    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    messageDiv.appendChild(messageContent);

    if (sender === 'user') {
        // User messages are plain text
        const paragraph = document.createElement('p');
        paragraph.textContent = text;
        messageContent.appendChild(paragraph);

        // Add image if provided
        if (imageSrc || (document.getElementById('image-preview-container').style.display === 'block')) {
            const img = document.createElement('img');
            // Use lazy loading for images
            img.dataset.src = imageSrc || document.getElementById('image-preview').src;
            img.alt = document.getElementById('image-preview').alt || "Uploaded image";
            img.className = "message-image";

            // Use a placeholder initially
            img.src = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1 1'%3E%3C/svg%3E";

            messageContent.appendChild(img);

            // Observe the image for lazy loading
            imageObserver.observe(img);
        }

        // Set a static title
        if (!titleSet) {
            chatTitle.textContent = "Malva Chat";
            titleSet = true;
        }

        // Store this as lastUserMessage to keep track of where to insert assistant responses
        lastUserMessage = messageDiv;
    } else {
        // Assistant messages may contain markdown
        const markdownDiv = document.createElement('div');
        markdownDiv.className = 'markdown';

        // Use optimized markdown formatting
        markdownDiv.innerHTML = formatMarkdownOptimized(text);
        messageContent.appendChild(markdownDiv);
    }

    // If this is an assistant message, insert after the last user message if available
    if (sender === 'assistant' && lastUserMessage) {
        lastUserMessage.insertAdjacentElement('afterend', messageDiv);
    } else {
        // Otherwise just append to the end
        messagesContainer.appendChild(messageDiv);
    }

    // Use optimized scrolling
    requestAnimationFrame(() => {
        smoothScrollToBottom(messagesContainer);
    });

    return messageDiv;
}

// Optimized font size change
function changeFontSize(direction) {
    requestAnimationFrame(() => {
        if (direction === 'increase' && fontSizeLevel < 150) {
            fontSizeLevel += 10;
        } else if (direction === 'decrease' && fontSizeLevel > 70) {
            fontSizeLevel -= 10;
        } else {
            return; // No change needed
        }

        // Apply the change
        document.documentElement.style.setProperty('--font-size', `${fontSizeLevel / 100}em`);

        // Announce the change
        announce(`Font size ${direction}d to ${fontSizeLevel} percent`);
    });
}

// Send text message
function sendTextMessage() {
    const text = userInput.value.trim();
    if (!text) {
        announce('Please enter a message', true);
        return;
    }

    // Add user message to chat first
    lastUserMessage = addMessage(text, 'user');

    // Reset any previous streaming message
    currentStreamingMessage = null;

    // Prepare data to send
    const data = {
        type: 'text',
        content: text,
        model: modelSelect.value,
        lang_code: languageSelect.value,
        voice: voiceSelect.value,
        tts_enabled: ttsToggle.checked
    };

    // Add image if available
    const imageContainer = document.getElementById('image-preview-container');
    if (imageContainer.style.display === 'block') {
        data.type = 'multimodal';
        data.text = text;
        data.image = document.getElementById('image-preview').src;
    }

    // Send via WebSocket
    if (ws && ws.readyState === WebSocket.OPEN) {
        // Create the processing message after the user message
        createProcessingMessage('Processing your message...');

        // Send the data
        ws.send(JSON.stringify(data));

        // Clear input
        userInput.value = '';
        userInput.style.height = 'auto';
        imageContainer.style.display = 'none';
        document.getElementById('image-preview').src = '';
        // Remove the remove button if it exists
        const removeBtn = document.getElementById('remove-image-btn');
        if (removeBtn) removeBtn.remove();
    } else {
        announce('Connection lost. Reconnecting...', true);
        connectWebSocket();
    }
}

// Start recording voice
function startRecording() {
    if (!navigator.mediaDevices) {
        announce('Voice recording is not supported in your browser', true);
        return;
    }

    navigator.mediaDevices.getUserMedia({ audio: {
        echoCancellation: true,
        noiseSuppression: true,
        channelCount: 1,
        sampleRate: 16000
    }})
        .then(stream => {
            announce('Recording started');
            micBtn.classList.add('btn-recording');
            micBtn.setAttribute('aria-label', 'Stop recording');
            isRecording = true;

            // Initialize MediaRecorder with WAV format
            // Note: Using webm format as it has better browser support
            // and will be converted properly on the server
            const options = {
                mimeType: 'audio/webm',
                audioBitsPerSecond: 128000
            };

            try {
                mediaRecorder = new MediaRecorder(stream, options);
            } catch (e) {
                // Fallback if preferred format not supported
                console.log('Preferred format not supported, using default format');
                mediaRecorder = new MediaRecorder(stream);
            }

            audioChunks = [];

            mediaRecorder.ondataavailable = (event) => {
                audioChunks.push(event.data);
            };

            mediaRecorder.onstop = () => {
                // Create audio blob
                const audioBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType });
                console.log('Audio recorded with MIME type:', mediaRecorder.mimeType);

                // Convert to base64
                const reader = new FileReader();
                reader.onloadend = () => {
                    const base64Audio = reader.result;

                    // Reset any previous streaming message
                    currentStreamingMessage = null;

                    // Send to server
                    if (ws && ws.readyState === WebSocket.OPEN) {
                        // Create processing message first
                        createProcessingMessage('Processing your voice...');

                        ws.send(JSON.stringify({
                            type: 'audio',
                            content: base64Audio,
                            model: modelSelect.value,
                            lang_code: languageSelect.value,
                            voice: voiceSelect.value,
                            tts_enabled: ttsToggle.checked
                        }));
                    } else {
                        announce('Connection lost. Reconnecting...', true);
                        connectWebSocket();
                    }
                };

                reader.readAsDataURL(audioBlob);
            };

            // Start recording with 100ms time slices
            mediaRecorder.start(100);
        })
        .catch(error => {
            console.error('Microphone error:', error);
            announce('Could not access microphone. Please check permissions.', true);
        });
}

// Stop recording
function stopRecording() {
    if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
        isRecording = false;
        micBtn.classList.remove('btn-recording');
        micBtn.setAttribute('aria-label', 'Record voice message');
        announce('Recording stopped, processing voice...');
    }
}

// Toggle recording
function toggleRecording() {
    if (isRecording) {
        stopRecording();
    } else {
        startRecording();
    }
}

// Handle image upload with accessibility improvements
function handleImageUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    // Check file type and size
    if (!file.type.startsWith('image/')) {
        announce('Please select an image file', true);
        return;
    }

    // Use FileReader API efficiently
    const reader = new FileReader();
    reader.onload = (e) => {
        const imagePreview = document.getElementById('image-preview');
        imagePreview.src = e.target.result;

        // Show the container instead of just the image
        const container = document.getElementById('image-preview-container');
        container.style.display = 'block';

        // Add remove button
        addImageRemoveButton();

        // Show alt text dialog for accessibility
        showAltTextDialog(file.name);
    };

    reader.readAsDataURL(file);
}

// Clear conversation with focus management
function clearConversation() {
    setFocusAfterOperation(userInput, () => {
        // Clear all messages
        messagesContainer.innerHTML = '';

        // Add initial message
        const initialMessage = document.createElement('div');
        initialMessage.className = 'message assistant-message';
        initialMessage.innerHTML = `
            <div class="message-avatar">M</div>
            <div class="message-content">
                <div class="markdown">
                    <p>Hello! I'm Malva, your accessible AI companion. I can understand text, voice, and images. How can I help you today?</p>
                </div>
            </div>
        `;
        messagesContainer.appendChild(initialMessage);

        // Clear input and image preview
        userInput.value = '';
        userInput.style.height = 'auto';

        // Remove uploaded image and button
        const container = document.getElementById('image-preview-container');
        container.style.display = 'none';
        document.getElementById('image-preview').src = '';

        // Reset title
        chatTitle.textContent = "Malva Chat";
        titleSet = false;

        // Reset streaming message
        currentStreamingMessage = null;
        currentProcessingMessage = null;
        lastUserMessage = null;

        // Clear caches
        markdownCache.clear();

        announce('Conversation cleared. Focus returned to message input.');
    });
}

// Improved screen reader announcement
function announce(message, assertive = false) {
    if (assertive) {
        ariaAlert.textContent = message;
    } else {
        ariaStatus.textContent = message;
    }
    console.log('Announced:', message, assertive ? '(assertive)' : '(polite)');
}

// Add metrics to the last assistant message
function addMetricsToLastMessage(messageTime) {
    // Store the time in the current streaming message
    if (currentStreamingMessage) {
        let metrics = currentStreamingMessage.querySelector('.metrics');
        if (!metrics) {
            metrics = document.createElement('div');
            metrics.className = 'metrics';
            currentStreamingMessage.querySelector('.message-content').appendChild(metrics);
        }

        // Store the response time for later use with audio
        metrics.dataset.responseTime = messageTime;

        metrics.innerHTML = `Response generated in ${messageTime}s`;
    }
}

// Set up event listeners
document.addEventListener('DOMContentLoaded', () => {
    // Check if image preview container exists, if not create it
    if (!document.getElementById('image-preview-container')) {
        const imagePreview = document.getElementById('image-preview');
        if (imagePreview) {
            const container = document.createElement('div');
            container.id = 'image-preview-container';
            container.style.display = 'none';
            container.style.position = 'relative';
            container.style.marginTop = '0.75rem';

            // Replace the image preview with the container + image
            const parent = imagePreview.parentNode;
            parent.insertBefore(container, imagePreview);
            container.appendChild(imagePreview);
        }
    }

    // Create better ARIA live regions
    ariaStatus = document.createElement('div');
    ariaStatus.setAttribute('aria-live', 'polite');
    ariaStatus.className = 'sr-only';
    document.body.appendChild(ariaStatus);

    ariaAlert = document.createElement('div');
    ariaAlert.setAttribute('aria-live', 'assertive');
    ariaAlert.className = 'sr-only';
    document.body.appendChild(ariaAlert);

    // Connect WebSocket
    connectWebSocket();

    // Set up language change listener
    languageSelect.addEventListener('change', () => {
        updateVoiceDropdown(languageSelect.value);
    });

    // Set initial voices
    updateVoiceDropdown('a');

    // Sidebar hover handlers
    sidebarTrigger.addEventListener('mouseenter', showSidebar);
    sidebarIndicator.addEventListener('mouseenter', showSidebar);
    sidebar.addEventListener('mouseleave', hideSidebar);
    sidebarToggle.addEventListener('click', () => {
        hideSidebar();
    });

    // Button click handlers with passive event listeners where appropriate
    sendBtn.addEventListener('click', sendTextMessage, { passive: true });
    micBtn.addEventListener('click', toggleRecording, { passive: true });
    imageBtn.addEventListener('click', () => imageUpload.click(), { passive: true });
    clearBtn.addEventListener('click', clearConversation, { passive: true });
    imageUpload.addEventListener('change', handleImageUpload, { passive: true });
    increaseFontBtn.addEventListener('click', () => changeFontSize('increase'), { passive: true });
    decreaseFontBtn.addEventListener('click', () => changeFontSize('decrease'), { passive: true });
    themeToggle.addEventListener('click', toggleTheme, { passive: true });
    toggleContrast.addEventListener('change', toggleHighContrast, { passive: true });

    // Input field handler
    userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendTextMessage();
        }
    });

    // Optimized textarea resize
    const resizeTextarea = debounce(() => {
        userInput.style.height = 'auto';
        userInput.style.height = Math.min(200, userInput.scrollHeight) + 'px';
    }, 100);

    userInput.addEventListener('input', resizeTextarea, { passive: true });

    // Optimized window resize handler
    window.addEventListener('resize', debounce(() => {
        if (userInput.value) resizeTextarea();
    }, 200), { passive: true });

    // Konami code listener
    document.addEventListener('keydown', (e) => {
        processKonamiCode(e.key);

        // Keyboard shortcuts
        // Alt+S: Send message
        if (e.altKey && e.key === 's') {
            e.preventDefault();
            sendTextMessage();
        }

        // Alt+V: Toggle voice recording
        if (e.altKey && e.key === 'v') {
            e.preventDefault();
            toggleRecording();
        }

        // Alt+I: Open image upload
        if (e.altKey && e.key === 'i') {
            e.preventDefault();
            imageUpload.click();
        }

        // Alt+C: Clear conversation
        if (e.altKey && e.key === 'c') {
            e.preventDefault();
            clearConversation();
        }

        // Alt+M: Toggle sidebar
        if (e.altKey && e.key === 'm') {
            e.preventDefault();
            toggleSidebar();
        }

        // Alt+T: Toggle theme
        if (e.altKey && e.key === 't') {
            e.preventDefault();
            toggleTheme();
        }

        // Escape: Stop recording if active
        if (e.key === 'Escape' && isRecording) {
            e.preventDefault();
            stopRecording();
        }
    });

    // Easter egg trigger with optimized animation
    easterEggTrigger.addEventListener('click', () => {
        const rects = easterEggTrigger.getBoundingClientRect();

        // Create sparkles in batches for better performance
        for (let i = 0; i < 5; i++) {
            setTimeout(() => {
                requestAnimationFrame(() => {
                    createSparkle(
                        rects.left + rects.width / 2 + (Math.random() * 40 - 20),
                        rects.top + rects.height / 2 + (Math.random() * 40 - 20)
                    );
                });
            }, i * 100);
        }
    });
});
