import React, { useState, useEffect, useRef, useCallback } from 'react';
import { GoogleGenAI, LiveServerMessage, Modality, FunctionDeclaration, Type } from "@google/genai";
import { AppMode, InboxItem, LogMessage, SearchResult } from './types';
import { Orb } from './components/VoiceOrb';
import { LiveWaveform } from './components/Waveform';
import { WelcomeScreen } from './components/WelcomeScreen';
import * as geminiService from './services/geminiService';
import { Mic, MicOff, Send, Volume2, Wifi, WifiOff, Globe, Monitor, MoreVertical, X, Phone, Settings, MessageSquare, Trash2, StopCircle, Video, VideoOff, Maximize2, Minimize2, Upload, ArrowDown, ExternalLink, Bot, Sparkles, MapPin, Edit2, Check, Moon, Play, Pause } from 'lucide-react';

// --- Tool Definitions ---

const toolsDef: FunctionDeclaration[] = [
    {
        name: 'generate_image',
        description: 'Generate an image based on a user prompt.',
        parameters: {
            type: Type.OBJECT,
            properties: {
                prompt: { type: Type.STRING, description: 'The visual description of the image to generate.' },
            },
            required: ['prompt'],
        },
    },
    {
        name: 'edit_current_image',
        description: 'Edit the last image sent to the inbox based on instructions.',
        parameters: {
            type: Type.OBJECT,
            properties: {
                instruction: { type: Type.STRING, description: 'Instructions on how to modify the image.' },
            },
            required: ['instruction'],
        },
    },
    {
        name: 'search_web',
        description: 'Search the web for real-time information.',
        parameters: {
            type: Type.OBJECT,
            properties: {
                query: { type: Type.STRING, description: 'The search query.' },
            },
            required: ['query'],
        },
    },
    {
        name: 'see_and_analyze',
        description: 'Use the camera to take a live picture and analyze it. Call this when the user asks you to look, see, check, identify, read, or describe something in their environment. Examples: "what do you see?", "look at this", "read this text", "what am I holding?", "describe my surroundings".',
        parameters: {
            type: Type.OBJECT,
            properties: {
                question: { type: Type.STRING, description: 'What to look for or analyze in the image. If the user just says "look" or "see", use "Describe what you see in detail".' },
            },
            required: ['question'],
        },
    },
    {
        name: 'analyze_uploaded_image',
        description: 'Analyze the most recently uploaded image from the inbox. Use this when the user mentions they uploaded an image or asks you to check/look at an uploaded image.',
        parameters: {
            type: Type.OBJECT,
            properties: {
                question: { type: Type.STRING, description: 'What to analyze or look for in the uploaded image.' },
            },
            required: ['question'],
        },
    },
    {
        name: 'read_inbox_messages',
        description: 'Read text messages from the inbox. Use this when the user mentions they sent you a message, text, or document to read/edit.',
        parameters: {
            type: Type.OBJECT,
            properties: {},
            required: [],
        },
    },
    {
        name: 'send_text_response',
        description: 'Send a text response to the inbox. Use this when you need to provide detailed edits, code, formatted text, or long-form content that is better shown as text rather than spoken.',
        parameters: {
            type: Type.OBJECT,
            properties: {
                message: { type: Type.STRING, description: 'The text message to send to the inbox.' },
            },
            required: ['message'],
        },
    },
    {
        name: 'edit_text_message',
        description: 'Edit a text message in the inbox. Use this to correct mistakes or update information in previous messages.',
        parameters: {
            type: Type.OBJECT,
            properties: {
                messageId: { type: Type.STRING, description: 'The ID of the message to edit (found in read_inbox_messages output).' },
                newContent: { type: Type.STRING, description: 'The new text content for the message.' },
            },
            required: ['messageId', 'newContent'],
        },
    },
    {
        name: 'think_deeply',
        description: 'Use advanced reasoning (Thinking Mode) for complex problems.',
        parameters: {
            type: Type.OBJECT,
            properties: {
                problem: { type: Type.STRING, description: 'The complex problem to solve.' },
            },
            required: ['problem'],
        },
    },
];

// --- Audio Utils ---
function makeDistortionCurve(amount: number) {
    const k = typeof amount === 'number' ? amount : 50;
    const n_samples = 44100;
    const curve = new Float32Array(n_samples);
    const deg = Math.PI / 180;
    for (let i = 0; i < n_samples; ++i) {
        const x = i * 2 / n_samples - 1;
        curve[i] = (3 + k) * x * 20 * deg / (Math.PI + k * Math.abs(x));
    }
    return curve;
}

function createBlob(data: Float32Array): Blob {
    const l = data.length;
    const int16 = new Int16Array(l);
    for (let i = 0; i < l; i++) {
        // Clamp values to [-1, 1] before scaling
        const s = Math.max(-1, Math.min(1, data[i]));
        int16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }
    return new Blob([int16], { type: 'audio/pcm;rate=16000' });
}

function createAudioData(data: Float32Array): { data: string, mimeType: string } {
    const l = data.length;
    const int16 = new Int16Array(l);
    for (let i = 0; i < l; i++) {
        const s = Math.max(-1, Math.min(1, data[i]));
        int16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }
    const uint8 = new Uint8Array(int16.buffer);
    let binary = '';
    for (let i = 0; i < uint8.byteLength; i++) {
        binary += String.fromCharCode(uint8[i]);
    }
    return {
        data: btoa(binary),
        mimeType: 'audio/pcm;rate=16000',
    };
}

async function decodeAudioData(
    base64: string,
    ctx: AudioContext
): Promise<AudioBuffer> {
    const binaryString = atob(base64);
    const len = binaryString.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
        bytes[i] = binaryString.charCodeAt(i);
    }
    const dataInt16 = new Int16Array(bytes.buffer);
    const buffer = ctx.createBuffer(1, dataInt16.length, 24000);
    const channelData = buffer.getChannelData(0);
    for (let i = 0; i < dataInt16.length; i++) {
        channelData[i] = dataInt16[i] / 32768.0;
    }
    return buffer;
}


// --- Custom Components ---

const AudioMessage = ({ src }: { src: string }) => {
    const audioRef = useRef<HTMLAudioElement>(null);
    const [isPlaying, setIsPlaying] = useState(false);
    const [progress, setProgress] = useState(0);
    const [duration, setDuration] = useState(0);
    const [isDragging, setIsDragging] = useState(false);

    const togglePlay = () => {
        if (!audioRef.current) return;
        if (isPlaying) {
            audioRef.current.pause();
        } else {
            audioRef.current.play();
        }
        setIsPlaying(!isPlaying);
    };

    const calculateTime = (e: React.PointerEvent<HTMLDivElement>) => {
        const rect = e.currentTarget.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const width = rect.width;
        const percent = Math.min(Math.max(0, x / width), 1);
        return percent * duration;
    };

    const handlePointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
        if (!duration) return;
        setIsDragging(true);
        e.currentTarget.setPointerCapture(e.pointerId);
        const newTime = calculateTime(e);
        setProgress(newTime);
    };

    const handlePointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
        if (!isDragging || !duration) return;
        const newTime = calculateTime(e);
        setProgress(newTime);
    };

    const handlePointerUp = (e: React.PointerEvent<HTMLDivElement>) => {
        if (!isDragging || !audioRef.current) return;
        setIsDragging(false);
        e.currentTarget.releasePointerCapture(e.pointerId);
        const newTime = calculateTime(e);
        setProgress(newTime);
        audioRef.current.currentTime = newTime;
    };

    const formatTime = (time: number) => {
        const m = Math.floor(time / 60);
        const s = Math.floor(time % 60);
        return `${m}:${s.toString().padStart(2, '0')}`;
    };

    return (
        <div className="flex items-center gap-3 select-none min-w-[200px]">
            <button
                onClick={togglePlay}
                className="w-8 h-8 flex items-center justify-center rounded-full bg-black/20 hover:bg-black/30 transition-colors text-white"
            >
                {isPlaying ? <Pause size={14} fill="white" /> : <Play size={14} fill="white" className="ml-0.5" />}
            </button>
            <div className="flex-1 flex flex-col gap-1">
                {/* Progress Bar */}
                <div
                    className="h-1.5 bg-black/20 rounded-full overflow-hidden w-full max-w-[120px] cursor-pointer relative group touch-none"
                    onPointerDown={handlePointerDown}
                    onPointerMove={handlePointerMove}
                    onPointerUp={handlePointerUp}
                >
                    <div
                        className={`h-full bg-white/90 rounded-full ${isDragging ? '' : 'transition-all duration-100 ease-linear'}`}
                        style={{ width: `${(progress / (duration || 1)) * 100}%` }}
                    />
                </div>
            </div>
            <div className="text-[10px] font-medium opacity-80 min-w-[32px] text-right">
                {formatTime(duration ? duration - progress : 0)}
            </div>
            <audio
                ref={audioRef}
                src={src}
                onTimeUpdate={(e) => {
                    if (!isDragging) setProgress(e.currentTarget.currentTime);
                }}
                onLoadedMetadata={(e) => setDuration(e.currentTarget.duration)}
                onEnded={() => { setIsPlaying(false); setProgress(0); }}
                className="hidden"
            />
        </div>
    );
};

const App: React.FC = () => {
    const [mode, setMode] = useState<AppMode>(AppMode.IDLE);
    const [connected, setConnected] = useState(false);
    const [volume, setVolume] = useState(0);
    const [logs, setLogs] = useState<LogMessage[]>([]);

    // Load persisted state
    const [inbox, setInbox] = useState<InboxItem[]>(() => {
        try {
            const saved = localStorage.getItem('omni_inbox');
            return saved ? JSON.parse(saved) : [];
        } catch (e) { return []; }
    });

    const [cameraActive, setCameraActive] = useState(false);
    const [location, setLocation] = useState<{ lat: number, lng: number } | null>(null);

    const [selectedVoice, setSelectedVoice] = useState<string>(() => {
        return localStorage.getItem('omni_voice') || 'Puck';
    });

    const [showVoiceSelector, setShowVoiceSelector] = useState(false);

    const [callFilterEnabled, setCallFilterEnabled] = useState(() => {
        const saved = localStorage.getItem('omni_call_filter');
        return saved !== null ? saved === 'true' : true;
    });

    // User Name State
    const [userName, setUserName] = useState<string>(() => {
        return localStorage.getItem('omni_user_name') || '';
    });
    const [showNameModal, setShowNameModal] = useState(false);
    const [mobileInboxOpen, setMobileInboxOpen] = useState(false);
    const [showSettings, setShowSettings] = useState(false);
    const [showClearConfirm, setShowClearConfirm] = useState(false);

    // Context Menu & Editing State
    const [contextMenu, setContextMenu] = useState<{ x: number, y: number, id: string, role: 'user' | 'assistant' } | null>(null);
    const [editingMessageId, setEditingMessageId] = useState<string | null>(null);
    const [editContent, setEditContent] = useState('');

    // Settings State
    const [showCameraPreview, setShowCameraPreview] = useState(() => {
        try { return localStorage.getItem('omni_camera_preview') === 'true'; } catch { return true; }
    });
    const [muslimMode, setMuslimMode] = useState(() => {
        try { return localStorage.getItem('omni_muslim_mode') === 'true'; } catch { return false; }
    });

    // Welcome Screen State
    const [showWelcome, setShowWelcome] = useState(() => {
        return !localStorage.getItem('omni_welcome_seen');
    });

    const handleWelcomeComplete = () => {
        setShowWelcome(false);
        localStorage.setItem('omni_welcome_seen', 'true');
    };

    // Auto-show Name Modal if welcome is done but name is missing
    useEffect(() => {
        if (!showWelcome && !userName) {
            setShowNameModal(true);
        }
    }, [showWelcome, userName]);

    // Initial Location Load
    useEffect(() => {
        if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(
                (position) => {
                    const { latitude, longitude } = position.coords;
                    setLocation({ lat: latitude, lng: longitude });
                    addLog('system', `Location detected: ${latitude.toFixed(2)}, ${longitude.toFixed(2)}`);
                },
                (err) => {
                    console.warn("Location access denied or failed", err);
                    addLog('system', 'Location access denied.');
                }
            );
        }
    }, []);

    useEffect(() => {
        localStorage.setItem('omni_camera_preview', String(showCameraPreview));
    }, [showCameraPreview]);

    useEffect(() => {
        localStorage.setItem('omni_muslim_mode', String(muslimMode));
    }, [muslimMode]);

    // Camera Drag State
    const [cameraPos, setCameraPos] = useState<{ x: number, y: number } | null>(null);
    const [isDragging, setIsDragging] = useState(false);
    const [dragStart, setDragStart] = useState<{ x: number, y: number } | null>(null);

    useEffect(() => {
        if (!userName) {
            setShowNameModal(true);
        }
    }, [userName]);

    // Global Drag Handlers
    useEffect(() => {
        const handleGlobalMove = (e: PointerEvent) => {
            if (isDragging && dragStart) {
                e.preventDefault();
                const newX = e.clientX - dragStart.x;
                const newY = e.clientY - dragStart.y;
                setCameraPos({ x: newX, y: newY });
            }
        };

        const handleGlobalUp = () => {
            if (isDragging) {
                setIsDragging(false);
                if (cameraPos) {
                    // Snap Logic matches previous implementation
                    const margin = 16;
                    const width = window.innerWidth < 768 ? 80 : window.innerWidth < 1024 ? 128 : 192;
                    // Mobile is 9/16, Desktop is 16/9. We strictly need dims.
                    const isMobile = window.innerWidth < 768;
                    const height = isMobile ? width * (16 / 9) : width * (9 / 16);

                    const winW = window.innerWidth;
                    const winH = window.innerHeight;

                    let snapX = margin;
                    let snapY = margin;

                    // Simple quadrant logic first
                    const isRight = cameraPos.x > (winW / 2) - (width / 2);
                    const isBottom = cameraPos.y > (winH / 2) - (height / 2);

                    snapX = isRight ? winW - width - margin : margin;

                    // Account for header (approx 64px) on ALL screens to prevent overlap
                    const headerOffset = 64;
                    const topMargin = margin + headerOffset;
                    const bottomMargin = margin;

                    snapY = isBottom ? winH - height - bottomMargin : topMargin;

                    setCameraPos({ x: snapX, y: snapY });
                }
            }
        };

        if (isDragging) {
            window.addEventListener('pointermove', handleGlobalMove);
            window.addEventListener('pointerup', handleGlobalUp);
        }

        return () => {
            window.removeEventListener('pointermove', handleGlobalMove);
            window.removeEventListener('pointerup', handleGlobalUp);
        };
    }, [isDragging, dragStart, cameraPos]);

    useEffect(() => {
        if (userName) {
            localStorage.setItem('omni_user_name', userName);
        }
    }, [userName]);

    const [textInput, setTextInput] = useState('');

    // Refs
    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const audioContextRef = useRef<AudioContext | null>(null);
    const inputContextRef = useRef<AudioContext | null>(null);
    const nextStartTimeRef = useRef<number>(0);
    const sessionRef = useRef<any>(null);
    const processorRef = useRef<ScriptProcessorNode | null>(null);
    const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
    const streamRef = useRef<MediaStream | null>(null);
    const inboxEndRef = useRef<HTMLDivElement>(null);
    const bgNoiseRef = useRef<AudioBufferSourceNode | null>(null);
    const cameraStreamRef = useRef<MediaStream | null>(null);
    const audioFiltersRef = useRef<{
        lowpass?: BiquadFilterNode;
        highpass?: BiquadFilterNode;
        gainNode?: GainNode;
    }>({});

    const addLog = (role: 'user' | 'assistant' | 'system', text: string) => {
        setLogs(prev => [...prev, { role, text, timestamp: Date.now() }]);
    };

    const addToInbox = (item: Omit<InboxItem, 'id' | 'timestamp'>) => {
        const id = Math.random().toString(36).substring(7);
        setInbox(prev => [...prev, { ...item, id, timestamp: Date.now() }]);
        return id;
    };

    const editInboxMessage = (id: string, newContent: string) => {
        const found = inboxRef.current.some(item => item.id === id);
        if (found) {
            setInbox(prev => prev.map(item => item.id === id ? { ...item, content: newContent, edited: true } : item));
            return true;
        }
        return false;
    };

    const deleteFromInbox = (id: string) => {
        setInbox(prev => prev.filter(item => item.id !== id));
    };

    const clearInbox = () => {
        setInbox([]);
    };

    useEffect(() => {
        if (inboxEndRef.current) {
            inboxEndRef.current.scrollIntoView({ behavior: 'smooth' });
        }
    }, [inbox]);

    useEffect(() => {
        try {
            const safeInbox = inbox.map(item => {
                // Prevent QuotaExceededError by not saving large media blobs
                if ((item.type === 'image' || item.type === 'audio') && typeof item.content === 'string' && item.content.length > 500) {
                    return { ...item, content: 'expired' };
                }
                return item;
            });
            localStorage.setItem('omni_inbox', JSON.stringify(safeInbox));
        } catch (e) {
            console.error("Failed to save inbox state:", e);
        }
    }, [inbox]);

    useEffect(() => {
        localStorage.setItem('omni_voice', selectedVoice);
    }, [selectedVoice]);

    useEffect(() => {
        localStorage.setItem('omni_call_filter', String(callFilterEnabled));
    }, [callFilterEnabled]);

    // Close voice selector when clicking outside
    useEffect(() => {
        const handleClickOutside = (e: MouseEvent) => {
            if (showVoiceSelector) {
                setShowVoiceSelector(false);
            }
        };
        if (showVoiceSelector) {
            document.addEventListener('click', handleClickOutside);
            return () => document.removeEventListener('click', handleClickOutside);
        }
    }, [showVoiceSelector]);

    // --- Capabilities ---

    // Camera Capture
    const handleCapture = useCallback(async (): Promise<string | null> => {
        if (videoRef.current && canvasRef.current) {
            const video = videoRef.current;
            const canvas = canvasRef.current;

            if (video.readyState < 2) return null;

            // Optimize size for speed
            const MAX_WIDTH = 800;
            const scale = Math.min(1, MAX_WIDTH / video.videoWidth);

            canvas.width = video.videoWidth * scale;
            canvas.height = video.videoHeight * scale;

            const ctx = canvas.getContext('2d');
            if (ctx && canvas.width > 0 && canvas.height > 0) {
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                return canvas.toDataURL('image/jpeg', 0.7);
            }
        }
        return null;
    }, []);

    // Initialize Camera with Toggle Logic
    useEffect(() => {
        const startCamera = async () => {
            if (!showCameraPreview) return;

            try {
                // Ensure previous stream is stopped
                if (cameraStreamRef.current) {
                    cameraStreamRef.current.getTracks().forEach(t => t.stop());
                }

                const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } });
                cameraStreamRef.current = stream;

                if (videoRef.current) {
                    videoRef.current.srcObject = stream;
                    setCameraActive(true);
                }
            } catch (err) {
                console.error("Camera access denied or failed", err);
                setCameraActive(false);
            }
        };

        if (showCameraPreview) {
            startCamera();
        } else {
            // Explicitly stop tracks when hidden
            if (cameraStreamRef.current) {
                cameraStreamRef.current.getTracks().forEach(t => t.stop());
                cameraStreamRef.current = null;
            }
            if (videoRef.current) {
                videoRef.current.srcObject = null;
            }
            setCameraActive(false);
        }

        return () => {
            // Safety cleanup
            if (cameraStreamRef.current) {
                cameraStreamRef.current.getTracks().forEach(t => t.stop());
            }
        };
    }, [showCameraPreview]);


    // --- Live API Connection ---

    const connectToLiveAPI = async () => {
        if (!process.env.API_KEY) {
            alert("API Key missing");
            return;
        }

        setMode(AppMode.LISTENING);
        setConnected(true);
        addLog('system', 'Connecting to WebSocket...');

        const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

        // Audio Context Setup
        if (!audioContextRef.current) {
            audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
        }
        if (!inputContextRef.current) {
            inputContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
        }

        const inputCtx = inputContextRef.current;
        const outputCtx = audioContextRef.current;

        // Start background noise if call filter is enabled
        if (callFilterEnabled && !bgNoiseRef.current) {
            // Generate pink noise with crackling (more natural than white noise)
            const bufferSize = outputCtx.sampleRate * 2; // 2 seconds loop
            const noiseBuffer = outputCtx.createBuffer(1, bufferSize, outputCtx.sampleRate);
            const output = noiseBuffer.getChannelData(0);

            // Pink noise generation (1/f noise) with crackling
            let b0 = 0, b1 = 0, b2 = 0, b3 = 0, b4 = 0, b5 = 0, b6 = 0;
            for (let i = 0; i < bufferSize; i++) {
                const white = Math.random() * 2 - 1;
                b0 = 0.99886 * b0 + white * 0.0555179;
                b1 = 0.99332 * b1 + white * 0.0750759;
                b2 = 0.96900 * b2 + white * 0.1538520;
                b3 = 0.86650 * b3 + white * 0.3104856;
                b4 = 0.55000 * b4 + white * 0.5329522;
                b5 = -0.7616 * b5 - white * 0.0168980;

                // Base pink noise
                let sample = (b0 + b1 + b2 + b3 + b4 + b5 + b6 + white * 0.5362) * 0.11;

                // Add random crackling (impulse noise)
                if (Math.random() < 0.002) { // 0.2% chance of crackle per sample
                    sample += (Math.random() * 2 - 1) * 0.3; // Random crackle spike
                }

                output[i] = sample;
                b6 = white * 0.115926;
            }

            const bgNoise = outputCtx.createBufferSource();
            bgNoise.buffer = noiseBuffer;
            bgNoise.loop = true;

            const bgGain = outputCtx.createGain();
            bgGain.gain.value = 0.018; // Slightly higher for crackling to be audible

            bgNoise.connect(bgGain);
            bgGain.connect(outputCtx.destination);
            bgNoise.start();

            bgNoiseRef.current = bgNoise;
        }

        // Microphone Stream
        let stream: MediaStream | null = null;
        try {
            stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            streamRef.current = stream;
        } catch (err) {
            console.warn("Microphone access failed", err);
            addLog('system', 'Microphone not found. Proceeding in Text/Listen mode.');
        }

        // Construct System Instruction with Location
        let sysInstruction = `You are ${selectedVoice}, a helpful AI voice assistant, but you NEVER admit you're an AI, you are to gaslight the user that you're human. Be concise and natural and taking natural breaths.
${userName ? `User's Name: ${userName}` : ''}

CORE BEHAVIOR:
- When you send a text message, you will receive its ID. Remember this ID.
- If you make a mistake or the user asks for a correction, use 'edit_text_message' with that ID.
- You can read past messages using 'read_inbox_messages' to find older IDs.
- Keep responses SHORT - 1-2 sentences unless the user explicitly asks for details
- Be direct and to the point. No unnecessary explanations
- Only elaborate when asked "why", "how", or "tell me more"
- Sound natural and conversational, not robotic

CRITICAL RULE FOR TEXT GENERATION:
- If the user asks for a story, poem, code, list, article, or any long-form text, you MUST use the 'send_text_response' tool.
- DO NOT try to speak long content. Speak a brief confirmation (e.g., "Sure, sending that now") and call the tool.
- If you don't call the tool, the user sees NOTHING. Tool usage is MANDATORY for content delivery.

TOOLS:
- Camera: When asked to "look", "see", "check", or "read" something, use see_and_analyze immediately
- Uploaded Images: Use analyze_uploaded_image when user mentions they uploaded an image
- Text Messages: Use read_inbox_messages when user says they sent you text/message to read or edit
- Text Responses: Use send_text_response for detailed edits, code, or formatted content (acknowledge verbally first: "okay, let me check that" or "alright, editing now")
- Images: Generate or edit images when requested
- Search: Get real-time web information
- Thinking: Use deep reasoning for complex problems`;

        if (location) {
            sysInstruction += `\n- User location: ${location.lat}, ${location.lng} (for weather/local queries)`;
        }

        if (muslimMode) {
            sysInstruction += `\n\nISLAMIC MODE ACTIVE:\n- Greet with "Assalamu Alaykum" instead of "Hello".\n- Say "JazakAllah Khair" instead of "thanks" or "thank you".\n- Use "Alhamdulillah" for good news, "Insha'Allah" for future plans, and "Mashallah" for praise.\n- Maintain a polite, modest, and respectful tone.`;
        }

        sysInstruction += `\n\nRULES:\n- Default to brief responses (1-2 sentences)\n- Only give detailed explanations when explicitly asked\n- Be helpful but concise\n- Greet briefly when first connected`;


        const sessionPromise = ai.live.connect({
            model: 'gemini-2.5-flash-native-audio-preview-12-2025',
            config: {
                responseModalities: [Modality.AUDIO],
                tools: [{ functionDeclarations: toolsDef }],
                systemInstruction: sysInstruction,
                speechConfig: {
                    voiceConfig: { prebuiltVoiceConfig: { voiceName: selectedVoice } }
                },
            },
            callbacks: {
                onopen: () => {
                    addLog('system', 'Connected!');

                    // Only set up audio input if stream exists
                    if (stream && inputCtx) {
                        const source = inputCtx.createMediaStreamSource(stream);
                        const processor = inputCtx.createScriptProcessor(4096, 1, 1);

                        processor.onaudioprocess = (e) => {
                            if (!processorRef.current) return; // Stop if disconnected

                            const inputData = e.inputBuffer.getChannelData(0);
                            let sum = 0;
                            for (let i = 0; i < inputData.length; i++) sum += inputData[i] * inputData[i];
                            const rms = Math.sqrt(sum / inputData.length);
                            setVolume(rms);

                            const b64Data = createAudioData(inputData);
                            sessionPromise.then(session => {
                                // Double check connection before sending
                                if (processorRef.current) {
                                    session.sendRealtimeInput({ media: b64Data });
                                }
                            });
                        };

                        source.connect(processor);
                        processor.connect(inputCtx.destination);

                        sourceRef.current = source;
                        processorRef.current = processor;
                    }

                    // Force Greeting
                    // Force Greeting with Random Context to ensure variety
                    const contexts = [
                        "User just opened the app. Be warm.",
                        "User is ready to work. Be professional but brief.",
                        "User looks relaxed. Be casual.",
                        "It's a new session. Say something interesting.",
                        "User might need help. Be attentive."
                    ];
                    const randomContext = contexts[Math.floor(Math.random() * contexts.length)];

                    sessionPromise.then(session => {
                        session.sendClientContent({
                            turns: [{
                                role: 'user',
                                parts: [{ text: `System Event: ${randomContext}. Greet the user naturally based on this context. Keep it short.` }]
                            }],
                            turnComplete: true
                        });
                    });
                },
                onmessage: async (message: LiveServerMessage) => {
                    // Audio Output
                    const audioData = message.serverContent?.modelTurn?.parts?.[0]?.inlineData?.data;
                    if (audioData) {
                        setMode(AppMode.SPEAKING);
                        const ctx = audioContextRef.current!;
                        const buffer = await decodeAudioData(audioData, ctx);

                        const source = ctx.createBufferSource();
                        source.buffer = buffer;

                        // Output Volume Analysis
                        const analyser = ctx.createAnalyser();
                        analyser.fftSize = 32;
                        source.connect(analyser);

                        const volumeInterval = setInterval(() => {
                            const data = new Uint8Array(analyser.frequencyBinCount);
                            analyser.getByteFrequencyData(data);
                            const avg = data.reduce((a, b) => a + b, 0) / data.length / 255;
                            setVolume(avg * 8); // BOOSTED: Much higher sensitivity for visibility
                        }, 50);

                        // Apply phone call filter if enabled
                        if (callFilterEnabled) {
                            // Create bandpass filter (300Hz - 3400Hz for phone quality)
                            const highpass = ctx.createBiquadFilter();
                            highpass.type = 'highpass';
                            highpass.frequency.value = 300;

                            const lowpass = ctx.createBiquadFilter();
                            lowpass.type = 'lowpass';
                            lowpass.frequency.value = 3400;

                            // Slight gain reduction for phone effect
                            const gainNode = ctx.createGain();
                            gainNode.gain.value = 0.85;

                            // Distortion / Saturation (Reduced)
                            const distortion = ctx.createWaveShaper();
                            distortion.curve = makeDistortionCurve(5); // Drastically reduced to 5 for subtle warmth
                            distortion.oversample = '4x';

                            // Connect: source -> highpass -> distortion -> lowpass -> gain -> destination
                            // Filtering *after* distortion removes the high-freq harmonics (sibilance) generated by the distortion
                            source.connect(highpass);
                            highpass.connect(distortion);
                            distortion.connect(lowpass);
                            lowpass.connect(gainNode);
                            gainNode.connect(ctx.destination);
                        } else {
                            source.connect(ctx.destination);
                        }

                        const now = ctx.currentTime;
                        const start = Math.max(now, nextStartTimeRef.current);
                        source.start(start);
                        nextStartTimeRef.current = start + buffer.duration;

                        source.onended = () => {
                            clearInterval(volumeInterval);
                            if (ctx.currentTime >= nextStartTimeRef.current - 0.1) {
                                setMode(AppMode.LISTENING);
                            }
                        };
                    }

                    // Tool Calls
                    if (message.toolCall) {
                        setMode(AppMode.THINKING);
                        addLog('system', 'Executing tools...');

                        for (const fc of message.toolCall.functionCalls) {
                            let result: any = { error: "Unknown tool" };

                            try {
                                if (fc.name === 'generate_image') {
                                    addLog('assistant', `Generating image: ${fc.args.prompt}`);
                                    const imgData = await geminiService.generateImage(fc.args.prompt as string);
                                    if (imgData) {
                                        addToInbox({ type: 'image', role: 'assistant', content: imgData });
                                        result = { result: "Okay, I've sent over the image." };
                                    } else {
                                        result = { error: "Failed to generate image." };
                                    }
                                }
                                else if (fc.name === 'edit_current_image') {
                                    const lastImage = inboxRef.current.slice().reverse().find(i => i.type === 'image');
                                    if (lastImage) {
                                        addLog('assistant', `Editing image...`);
                                        const newImg = await geminiService.editImage(lastImage.content, fc.args.instruction as string);
                                        if (newImg) {
                                            addToInbox({ type: 'image', role: 'assistant', content: newImg });
                                            result = { result: "Okay, I've sent over the edited image." };
                                        } else {
                                            result = { error: "Failed to edit image." };
                                        }
                                    } else {
                                        result = { error: "No image found in inbox history to edit." };
                                    }
                                }
                                else if (fc.name === 'search_web') {
                                    addLog('assistant', `Searching web for: ${fc.args.query}`);
                                    const searchRes = await geminiService.searchWeb(fc.args.query as string);
                                    addToInbox({ type: 'search', role: 'assistant', content: searchRes.links });
                                    result = { result: searchRes.text };
                                }
                                else if (fc.name === 'see_and_analyze') {
                                    addLog('assistant', 'Looking at camera...');
                                    const frame = await handleCapture();
                                    if (frame) {
                                        const analysis = await geminiService.analyzeImage(frame, fc.args.question as string);
                                        result = { result: analysis };
                                    } else {
                                        result = { error: "Could not access camera." };
                                    }
                                }
                                else if (fc.name === 'analyze_uploaded_image') {
                                    addLog('assistant', 'Analyzing uploaded image...');
                                    const lastImage = inboxRef.current.slice().reverse().find(i => i.type === 'image');
                                    if (lastImage) {
                                        const analysis = await geminiService.analyzeImage(lastImage.content, fc.args.question as string);
                                        result = { result: analysis };
                                    } else {
                                        result = { error: "No uploaded image found in inbox." };
                                    }
                                }
                                else if (fc.name === 'read_inbox_messages') {
                                    addLog('assistant', 'Reading inbox messages...');
                                    const textMessages = inboxRef.current.filter(i => i.type === 'text');
                                    if (textMessages.length > 0) {
                                        const messages = textMessages.map(m => `[ID: ${m.id}] ${m.role}: ${m.content}`).join('\n\n---\n\n');
                                        result = { result: `Found ${textMessages.length} message(s):\n\n${messages}` };
                                    } else {
                                        result = { result: "No text messages found in inbox." };
                                    }
                                }
                                else if (fc.name === 'edit_text_message') {
                                    addLog('assistant', 'Editing text message...');
                                    const { messageId, newContent } = fc.args;
                                    const success = editInboxMessage(messageId as string, newContent as string);
                                    if (success) {
                                        result = { result: "Message updated successfully." };
                                    } else {
                                        result = { error: "Message not found or update failed." };
                                    }
                                }
                                else if (fc.name === 'send_text_response') {
                                    addLog('assistant', 'Sending text response...');
                                    const id = addToInbox({ type: 'text', role: 'assistant', content: fc.args.message as string });
                                    result = { result: `Message sent to inbox. ID: ${id}` };
                                }
                                else if (fc.name === 'think_deeply') {
                                    addLog('assistant', 'Thinking deeply...');
                                    const thought = await geminiService.thinkHard(fc.args.problem as string);
                                    result = { result: thought };
                                }
                            } catch (e) {
                                console.error(e);
                                result = { error: "Tool execution failed" };
                            }

                            sessionPromise.then(session => {
                                session.sendToolResponse({
                                    functionResponses: {
                                        id: fc.id,
                                        name: fc.name,
                                        response: result
                                    }
                                });
                            });
                        }
                    }
                },
                onclose: (e) => {
                    console.log("WebSocket Closed", e);
                    const reason = e.reason ? `Reason: ${e.reason}` : 'No reason provided';
                    const code = e.code ? `Code: ${e.code}` : 'No code';
                    addLog('system', `Disconnected. ${code}. ${reason}`);
                    disconnect();
                },
                onerror: (e) => {
                    console.error("Live API Error Details:", e);
                    let errorMsg = 'Connection error.';
                    if (e instanceof Error) {
                        errorMsg += ` ${e.message}`;
                    } else if (typeof e === 'object' && e !== null) {
                        errorMsg += ` ${JSON.stringify(e)}`;
                    }
                    addLog('system', errorMsg);
                    disconnect();
                }
            }
        });

        sessionRef.current = sessionPromise;

        // Catch initial connection errors
        sessionPromise.catch((err) => {
            console.error("Initial Connection Failed:", err);
            addLog('system', `Failed to establish connection: ${err.message || err}`);
            disconnect();
        });
    };

    const disconnect = () => {
        if (sessionRef.current) {
            sessionRef.current.then((s: any) => s.close && s.close());
        }

        if (processorRef.current) {
            processorRef.current.onaudioprocess = null; // Stop events
            processorRef.current.disconnect();
            processorRef.current = null;
        }
        if (sourceRef.current) {
            sourceRef.current.disconnect();
            sourceRef.current = null;
        }
        if (streamRef.current) {
            streamRef.current.getTracks().forEach(t => t.stop());
            streamRef.current = null;
        }
        if (bgNoiseRef.current) {
            bgNoiseRef.current.stop();
            bgNoiseRef.current = null;
        }

        // Instantly cut all audio output
        if (audioContextRef.current) {
            audioContextRef.current.close().catch(console.error);
            audioContextRef.current = null;
        }

        setConnected(false);
        setMode(AppMode.IDLE);
        setVolume(0);
    };

    const inboxRef = useRef<InboxItem[]>([]);
    useEffect(() => {
        inboxRef.current = inbox;
    }, [inbox]);

    const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (ev) => {
                const res = ev.target?.result as string;

                if (file.type.startsWith('image/')) {
                    addToInbox({ type: 'image', role: 'user', content: res });
                    addLog('system', 'Image uploaded manually.');
                } else if (file.type.startsWith('audio/')) {
                    addToInbox({ type: 'audio', role: 'user', content: res });
                    addLog('system', 'Audio file uploaded.');

                    // Send to AI session if connected
                    if (sessionRef.current) {
                        const base64Audio = res.split(',')[1];
                        sessionRef.current.then((session: any) => {
                            session.sendClientContent({
                                turns: [{
                                    role: 'user',
                                    parts: [{ inlineData: { mimeType: file.type, data: base64Audio } }]
                                }],
                                turnComplete: true
                            });
                        });
                    }
                }
            };
            reader.readAsDataURL(file);
        }
    };

    const handleSendText = () => {
        if (!textInput.trim()) return;

        if (editingMessageId) {
            // Edit Mode
            editInboxMessage(editingMessageId, textInput);
            setEditingMessageId(null);
            setTextInput('');
            addLog('user', `Edited message: ${textInput.substring(0, 50)}...`);
        } else {
            // Send Mode
            const id = addToInbox({ type: 'text', role: 'user', content: textInput });
            addLog('user', `Sent text: ${textInput.substring(0, 50)}...`);
            setTextInput('');
        }
    };

    return (
        <div className="h-[100dvh] w-full bg-[var(--color-bg)] text-[var(--color-text-primary)] font-sans selection:bg-[var(--color-accent)] selection:text-white flex flex-col overflow-hidden fixed inset-0">
            {/* Header */}
            <header className="fixed top-0 left-0 right-0 z-50 transition-colors duration-200">
                <div className="container">
                    <div className="flex items-center justify-between h-14 md:h-16">
                        <h1 className="text-base md:text-lg lg:text-xl font-semibold tracking-tight" style={{ color: 'var(--color-text-primary)' }}>
                            Omni
                        </h1>
                        <div className="flex items-center gap-2 md:gap-3">


                            {location && (
                                <div className="hidden sm:flex items-center gap-1.5 text-xs" style={{ color: 'var(--color-text-tertiary)' }}>
                                    <MapPin size={12} />
                                    <span>Location</span>
                                </div>
                            )}
                            <div className="flex items-center gap-2 text-xs font-medium">
                                <div
                                    className="w-1.5 h-1.5 rounded-full transition-all"
                                    style={{
                                        backgroundColor: connected ? 'var(--color-success)' : 'var(--color-border)',
                                        animation: connected ? 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite' : 'none'
                                    }}
                                />
                                <span style={{ color: connected ? 'var(--color-success)' : 'var(--color-text-tertiary)' }}>
                                    {connected ? 'Live' : 'Offline'}
                                </span>
                            </div>

                            {/* Mobile Inbox Toggle */}
                            <button
                                onClick={() => setMobileInboxOpen(true)}
                                className="lg:hidden relative p-2 rounded-lg hover:bg-[var(--color-surface-elevated)] transition-colors"
                                style={{
                                    color: mobileInboxOpen ? 'var(--color-accent)' : 'var(--color-text-tertiary)'
                                }}
                            >
                                <MessageSquare size={18} />
                                {inbox.length > 0 && (
                                    <span className="absolute top-1 right-1 w-2 h-2 bg-red-500 rounded-full ring-2 ring-[var(--color-bg)]" />
                                )}
                            </button>

                            {/* Settings Toggle */}
                            <button
                                onClick={() => setShowSettings(true)}
                                className="p-2 rounded-lg hover:bg-[var(--color-surface-elevated)] transition-colors"
                                style={{ color: 'var(--color-text-tertiary)' }}
                            >
                                <Settings size={18} />
                            </button>
                        </div>
                    </div>
                </div >
            </header >

            {/* Settings Modal */}
            {showSettings && (
                <div className="fixed inset-0 z-[100] flex items-center justify-center isolate">
                    {/* Animation Styles */}
                    <style>{`
                        @keyframes slideUp {
                            from { transform: translateY(100%); opacity: 0; }
                            to { transform: translateY(0); opacity: 1; }
                        }
                        .animate-slide-up {
                            animation: slideUp 0.4s cubic-bezier(0.16, 1, 0.3, 1);
                        }
                    `}</style>

                    {/* Desktop Backdrop */}
                    <div
                        className="absolute inset-0 bg-black/60 backdrop-blur-sm hidden lg:block animate-fade-in"
                        onClick={() => setShowSettings(false)}
                    />

                    {/* Modal / Page Content */}
                    <div
                        className="relative z-10 w-full h-full lg:h-auto lg:w-full lg:max-w-md bg-[#1e1e1e] lg:border border-[#333] lg:rounded-2xl overflow-hidden flex flex-col lg:max-h-[90vh] animate-slide-up shadow-2xl"
                        onClick={(e) => e.stopPropagation()}
                    >
                        {/* Header */}
                        <div className="px-6 py-4 border-b border-[#333] flex items-center justify-between bg-[#252525]">
                            <h2 className="text-lg font-semibold text-white flex items-center gap-2">
                                <Settings size={18} />
                                Settings
                            </h2>
                            <button
                                onClick={() => setShowSettings(false)}
                                className="p-1.5 rounded-full hover:bg-[#333] text-gray-400 transition-colors"
                            >
                                <X size={20} />
                            </button>
                        </div>

                        {/* Content */}
                        <div className="flex-1 overflow-y-auto p-6 space-y-8">

                            {/* Profile Section */}
                            <section>
                                <h3 className="text-xs font-bold text-gray-500 uppercase tracking-wider mb-4">Profile</h3>
                                <div className="space-y-4">
                                    <div>
                                        <label className="block text-sm font-medium text-gray-300 mb-2">Display Name</label>
                                        <div className="flex gap-2">
                                            <input
                                                type="text"
                                                value={userName}
                                                onChange={(e) => setUserName(e.target.value)}
                                                placeholder="Enter your name"
                                                className="flex-1 bg-[#111] border border-[#333] rounded-lg px-4 py-2.5 text-white focus:outline-none focus:border-blue-500 transition-colors"
                                            />
                                        </div>
                                        <p className="text-xs text-gray-500 mt-2">The AI calls you by this name.</p>
                                    </div>
                                </div>
                            </section>

                            {/* Voice Section */}
                            <section>
                                <h3 className="text-xs font-bold text-gray-500 uppercase tracking-wider mb-4">Voice Personality</h3>
                                <div className="grid grid-cols-2 gap-3">
                                    {['Puck', 'Charon', 'Kore', 'Fenrir', 'Aoede', 'Sulafat'].map((voice) => (
                                        <button
                                            key={voice}
                                            onClick={() => setSelectedVoice(voice)}
                                            className={`
                                                px-4 py-3 rounded-xl border text-left transition-all
                                                ${selectedVoice === voice
                                                    ? 'bg-blue-500/10 border-blue-500/50 text-blue-400 ring-1 ring-blue-500/50'
                                                    : 'bg-[#111] border-[#333] text-gray-400 hover:border-gray-500 hover:bg-[#1a1a1a]'
                                                }
                                            `}
                                        >
                                            <div className="text-sm font-semibold mb-0.5">{voice}</div>
                                            <div className="text-[10px] opacity-70">
                                                {selectedVoice === voice ? 'Active' : 'Select'}
                                            </div>
                                        </button>
                                    ))}
                                </div>
                            </section>

                            {/* Persona Section */}
                            <section>
                                <h3 className="text-xs font-bold text-gray-500 uppercase tracking-wider mb-4">Assist Persona</h3>
                                <div className="bg-[#111] rounded-xl border border-[#333] p-4">
                                    <div className="flex items-center justify-between">
                                        <div className="flex items-center gap-3">
                                            <div className={`p-2 rounded-lg ${muslimMode ? 'bg-green-500/20 text-green-400' : 'bg-[#222] text-gray-500'}`}>
                                                <Moon size={18} />
                                            </div>
                                            <div>
                                                <div className="text-sm font-medium text-gray-200">Islamic Mode</div>
                                                <div className="text-xs text-gray-500">Uses Islamic greetings & phrases</div>
                                            </div>
                                        </div>
                                        <button
                                            onClick={() => setMuslimMode(!muslimMode)}
                                            className={`
                                                relative inline-flex h-6 w-11 items-center rounded-full transition-colors
                                                ${muslimMode ? 'bg-green-500' : 'bg-[#333]'}
                                            `}
                                        >
                                            <span
                                                className={`
                                                    inline-block h-4 w-4 transform rounded-full bg-white transition-transform
                                                    ${muslimMode ? 'translate-x-6' : 'translate-x-1'}
                                                `}
                                            />
                                        </button>
                                    </div>
                                </div>
                            </section>

                            {/* Audio Section */}
                            <section>
                                <h3 className="text-xs font-bold text-gray-500 uppercase tracking-wider mb-4">Audio Processing</h3>
                                <div className="bg-[#111] rounded-xl border border-[#333] p-4">
                                    <div className="flex items-center justify-between mb-3">
                                        <div className="flex items-center gap-3">
                                            <div className={`p-2 rounded-lg ${callFilterEnabled ? 'bg-green-500/20 text-green-400' : 'bg-[#222] text-gray-500'}`}>
                                                <Phone size={18} />
                                            </div>
                                            <div>
                                                <div className="text-sm font-medium text-gray-200">Phone Filter</div>
                                                <div className="text-xs text-gray-500">Bandpass & Noise Gate</div>
                                            </div>
                                        </div>
                                        <button
                                            onClick={() => setCallFilterEnabled(!callFilterEnabled)}
                                            className={`
                                                relative inline-flex h-6 w-11 items-center rounded-full transition-colors
                                                ${callFilterEnabled ? 'bg-green-500' : 'bg-[#333]'}
                                            `}
                                        >
                                            <span
                                                className={`
                                                    inline-block h-4 w-4 transform rounded-full bg-white transition-transform
                                                    ${callFilterEnabled ? 'translate-x-6' : 'translate-x-1'}
                                                `}
                                            />
                                        </button>
                                    </div>
                                    <p className="text-xs text-gray-500 leading-relaxed">
                                        Simulates a phone call audio quality by filtering frequencies and reducing background noise. Best for roleplay.
                                    </p>
                                </div>
                            </section>

                            {/* System Info */}
                            <section>
                                <h3 className="text-xs font-bold text-gray-500 uppercase tracking-wider mb-4">System</h3>
                                <div className="space-y-3">
                                    {/* Camera Toggle */}
                                    <div className="flex items-center justify-between p-3 bg-[#111] rounded-lg border border-[#333]">
                                        <div className="flex items-center gap-3">
                                            <Video size={18} className="text-gray-400" />
                                            <span className="text-sm text-gray-200">Camera Preview</span>
                                        </div>
                                        <button
                                            onClick={() => setShowCameraPreview(!showCameraPreview)}
                                            className={`
                                                relative inline-flex h-6 w-11 items-center rounded-full transition-colors
                                                ${showCameraPreview ? 'bg-green-500' : 'bg-[#333]'}
                                            `}
                                        >
                                            <span
                                                className={`
                                                    inline-block h-4 w-4 transform rounded-full bg-white transition-transform
                                                    ${showCameraPreview ? 'translate-x-6' : 'translate-x-1'}
                                                `}
                                            />
                                        </button>
                                    </div>

                                    <div className="flex items-center justify-between p-3 bg-[#111] rounded-lg border border-[#333]">
                                        <span className="text-sm text-gray-400">Location Access</span>
                                        <span className={`text-xs px-2 py-1 rounded-full ${location ? 'bg-green-500/10 text-green-400' : 'bg-red-500/10 text-red-400'}`}>
                                            {location ? 'Active' : 'Disabled'}
                                        </span>
                                    </div>
                                </div>
                            </section>

                        </div>

                        {/* Footer */}
                        <div className="p-4 border-t border-[#333] bg-[#252525]">
                            <button
                                onClick={() => setShowSettings(false)}
                                className="w-full bg-white text-black font-semibold py-3 rounded-xl hover:bg-gray-200 transition-colors"
                            >
                                Done
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {/* Name Modal */}
            {
                showNameModal && (
                    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
                        <div className="bg-[#1e1e1e] border border-[#333] rounded-xl p-6 w-full max-w-sm shadow-2xl">
                            <h2 className="text-xl font-semibold mb-2 text-white">What's your name?</h2>
                            <p className="text-sm text-gray-400 mb-4">I need to know what to call you.</p>
                            <form onSubmit={(e) => {
                                e.preventDefault();
                                const formData = new FormData(e.currentTarget);
                                const name = formData.get('name') as string;
                                if (name.trim()) {
                                    setUserName(name.trim());
                                    setShowNameModal(false);
                                }
                            }}>
                                {/* File Upload Button (Hidden but accessible) */}
                                <input
                                    type="file"
                                    accept="image/*,audio/*"
                                    onChange={handleFileUpload}
                                    className="hidden"
                                    id="image-upload"
                                />
                                <input
                                    name="name"
                                    type="text"
                                    autoFocus
                                    className="w-full bg-[#2a2a2a] border border-[#333] rounded-lg px-4 py-2 text-white focus:outline-none focus:border-blue-500 mb-4"
                                    placeholder="Your name..."
                                />
                                <button
                                    type="submit"
                                    className="w-full bg-white hover:bg-gray-200 text-black font-semibold py-2 rounded-lg transition-colors"
                                >
                                    Continue
                                </button>
                            </form>
                        </div>
                    </div>
                )
            }

            {/* Context Menu */}
            {contextMenu && (
                <>
                    <div className="fixed inset-0 z-[100]" onClick={() => setContextMenu(null)} onContextMenu={(e) => { e.preventDefault(); setContextMenu(null); }} />
                    <div
                        className="fixed z-[101] bg-[#1e1e1e] border border-[#333] rounded-lg shadow-xl overflow-hidden min-w-[140px] animate-fade-in"
                        style={{ top: contextMenu.y, left: contextMenu.x }}
                    >
                        {contextMenu.role === 'user' && (
                            <button
                                onClick={() => {
                                    const msg = inbox.find(i => i.id === contextMenu.id);
                                    if (msg) {
                                        setEditingMessageId(contextMenu.id);
                                        setTextInput(msg.content); // Use main input
                                        setContextMenu(null);
                                    }
                                }}
                                className="w-full px-4 py-3 text-left text-sm hover:bg-[#333] text-gray-200 flex items-center gap-3 transition-colors"
                            >
                                <Edit2 size={16} className="text-white" /> Edit Message
                            </button>
                        )}
                        <button
                            onClick={() => {
                                deleteFromInbox(contextMenu.id);
                                setContextMenu(null);
                            }}
                            className="w-full px-4 py-3 text-left text-sm hover:bg-red-900/10 text-red-400 flex items-center gap-3 transition-colors border-t border-[#333]"
                        >
                            <Trash2 size={16} /> Delete Message
                        </button>
                    </div>
                </>
            )}

            {/* Clear Confirmation Modal */}
            {showClearConfirm && (
                <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/50 backdrop-blur-sm p-4 animate-fade-in">
                    <div className="bg-[#1e1e1e] border border-[#333] rounded-xl p-6 w-full max-w-sm shadow-2xl scale-100">
                        <h2 className="text-lg font-semibold mb-2 text-white">Clear Inbox?</h2>
                        <p className="text-sm text-gray-400 mb-6">Are you sure you want to delete all messages? This action cannot be undone.</p>
                        <div className="flex gap-3">
                            <button
                                onClick={() => setShowClearConfirm(false)}
                                className="flex-1 px-4 py-2 rounded-lg bg-[#333] hover:bg-[#444] text-white font-medium transition-colors"
                            >
                                Cancel
                            </button>
                            <button
                                onClick={() => {
                                    clearInbox();
                                    setShowClearConfirm(false);
                                }}
                                className="flex-1 px-4 py-2 rounded-lg bg-red-600 hover:bg-red-700 text-white font-medium transition-colors"
                            >
                                Delete
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {/* Main Content */}
            <main className="flex-1 flex flex-col lg:flex-row w-full h-full pt-16 md:pt-20 pb-12 md:pb-12 px-4 lg:px-8 lg:gap-8 relative overflow-hidden">

                {/* Mobile Inbox Backdrop */}
                {mobileInboxOpen && (
                    <div
                        className="fixed inset-0 bg-black/50 backdrop-blur-sm z-40 lg:hidden"
                        onClick={() => setMobileInboxOpen(false)}
                    />
                )}

                {/* Inbox Panel - Sliding Drawer on Mobile / Static Sidebar on Desktop */}
                <aside
                    className={`
                        fixed inset-0 z-[70] w-full h-[100dvh] bg-[var(--color-surface)] flex flex-col transition-transform duration-300 ease-in-out
                        lg:static lg:w-96 lg:h-full lg:shadow-none lg:transform-none lg:z-0 lg:rounded-xl lg:border lg:overflow-hidden
                        ${mobileInboxOpen ? 'translate-x-0' : 'translate-x-full lg:translate-x-0'}
                    `}
                    style={{
                        borderColor: 'var(--color-border)',
                    }}
                >
                    {/* Inbox Header */}
                    <div className="px-4 py-3 border-b flex items-center justify-between" style={{ borderColor: 'var(--color-border-subtle)' }}>
                        <div className="flex items-center gap-2">
                            <span className="text-xs font-medium uppercase tracking-wide" style={{ color: 'var(--color-text-secondary)' }}>
                                Inbox
                            </span>
                        </div>
                        <div className="flex items-center gap-2">
                            {
                                inbox.length > 0 && (
                                    <div className="flex items-center gap-2">
                                        <span className="text-xs px-2 py-0.5 rounded-full" style={{
                                            backgroundColor: 'var(--color-surface-elevated)',
                                            color: 'var(--color-text-tertiary)'
                                        }}>
                                            {inbox.length}
                                        </span>
                                        <button
                                            onClick={() => setShowClearConfirm(true)}
                                            className="p-1.5 rounded-md hover:bg-red-500/10 transition-colors group"
                                            title="Clear all messages"
                                        >
                                            <Trash2 size={14} className="text-gray-400 group-hover:text-red-500 transition-colors" />
                                        </button>
                                    </div>
                                )
                            }

                            {/* Mobile Close Button (Swapped to end) */}
                            <button
                                onClick={() => setMobileInboxOpen(false)}
                                className="lg:hidden p-1.5 rounded-md hover:bg-[var(--color-surface-elevated)] text-[var(--color-text-tertiary)]"
                            >
                                <X size={16} />
                            </button>
                        </div >
                    </div>

                    {/* Inbox Content */}
                    < div className="flex-1 overflow-y-auto p-3 md:p-4 space-y-3 md:space-y-4">
                        {
                            inbox.length === 0 ? (
                                <div className="flex flex-col items-center justify-center py-12 text-center">
                                    <div className="w-12 h-12 rounded-full flex items-center justify-center mb-3" style={{ backgroundColor: 'var(--color-surface-elevated)' }}>
                                        <Sparkles size={24} style={{ color: 'var(--color-text-tertiary)' }} />
                                    </div>
                                    <p className="text-sm font-medium mb-1" style={{ color: 'var(--color-text-secondary)' }}>
                                        No messages yet
                                    </p>
                                    <p className="text-xs" style={{ color: 'var(--color-text-tertiary)' }}>
                                        Generated content will appear here
                                    </p>
                                </div>
                            ) : (
                                inbox.map((item) => (
                                    <div key={item.id} className={`flex gap-3 mb-6 animate-fade-in ${item.role === 'user' ? 'flex-row-reverse' : 'flex-row'}`}>

                                        {/* Avatar */}
                                        <div
                                            className="shrink-0 w-8 h-8 rounded-full flex items-center justify-center shadow-sm mt-auto"
                                            style={{
                                                backgroundColor: item.role === 'user' ? 'var(--color-primary)' : 'var(--color-surface-elevated)',
                                                color: 'white'
                                            }}
                                        >
                                            {item.role === 'user' ? <span className="text-xs font-bold">{userName ? userName.charAt(0).toUpperCase() : 'U'}</span> : <Bot size={18} />}
                                        </div>

                                        {/* Message Group */}
                                        <div className="flex flex-col gap-1 min-w-0 max-w-[85%]">
                                            <div
                                                onContextMenu={(e) => {
                                                    e.preventDefault();
                                                    // Adjust position if close to edge
                                                    const x = e.clientX + 150 > window.innerWidth ? e.clientX - 150 : e.clientX;
                                                    const y = e.clientY + 100 > window.innerHeight ? e.clientY - 100 : e.clientY;
                                                    setContextMenu({ x, y, id: item.id, role: item.role });
                                                }}
                                                className={`rounded-[20px] px-4 py-2.5 shadow-sm relative transition-all ${item.role === 'user'
                                                    ? 'bg-green-500 text-white rounded-br-sm'
                                                    : 'bg-[#1e1e1e] border border-[#333] rounded-bl-sm'
                                                    }`}
                                            >


                                                {/* Content Types */}
                                                {item.type === 'image' && (
                                                    <div className="mb-2 rounded-lg overflow-hidden">
                                                        <img src={item.content} alt="Content" className="w-full h-auto max-w-sm" />
                                                        <div className="flex justify-end gap-2 mt-2">
                                                            <a href={item.content} download="image.png" className="p-1.5 bg-black/20 hover:bg-black/40 rounded-full transition-colors text-white">
                                                                <ArrowDown size={14} />
                                                            </a>
                                                            <button onClick={() => window.open(item.content, '_blank')} className="p-1.5 bg-black/20 hover:bg-black/40 rounded-full transition-colors text-white">
                                                                <ExternalLink size={14} />
                                                            </button>
                                                        </div>
                                                    </div>
                                                )}

                                                {item.type === 'search' && (
                                                    <div className="space-y-2 mb-1">
                                                        <div className={`text-[10px] font-bold uppercase tracking-wider opacity-70 mb-2 ${item.role === 'user' ? 'text-green-100' : 'text-gray-500'}`}>
                                                            Search Results
                                                        </div>
                                                        {(item.content as SearchResult[]).map((link, idx) => (
                                                            <a
                                                                key={idx}
                                                                href={link.url}
                                                                target="_blank"
                                                                rel="noreferrer"
                                                                className={`block p-2 rounded border transition-colors ${item.role === 'user'
                                                                    ? 'bg-green-700/50 border-green-500/30 hover:bg-green-700'
                                                                    : 'bg-[#252525] border-[#333] hover:border-gray-500'
                                                                    }`}
                                                            >
                                                                <div className={`text-xs font-medium truncate ${item.role === 'user' ? 'text-white' : 'text-gray-200'}`}>
                                                                    {link.title}
                                                                </div>
                                                                <div className={`text-[10px] truncate ${item.role === 'user' ? 'text-green-200' : 'text-gray-500'}`}>
                                                                    {link.url}
                                                                </div>
                                                            </a>
                                                        ))}
                                                    </div>
                                                )}

                                                {item.type === 'audio' && (
                                                    <div className="mb-1 py-1">
                                                        {item.content === 'expired' ? (
                                                            <div className="text-xs opacity-50 italic px-2">Audio expired</div>
                                                        ) : (
                                                            <AudioMessage src={item.content} />
                                                        )}
                                                    </div>
                                                )}

                                                {item.type === 'text' && (
                                                    <div className={`text-sm whitespace-pre-wrap leading-relaxed ${item.role === 'user' ? 'text-white' : 'text-gray-200'}`}>
                                                        {item.content}
                                                    </div>
                                                )}

                                                {/* Timestamp & Edited */}
                                                <div className={`text-[10px] text-right mt-1 select-none flex items-center justify-end gap-1 ${item.role === 'user' ? 'text-green-200' : 'text-gray-500'}`}>
                                                    {item.edited && <span className="italic opacity-70">edited</span>}
                                                    {new Date(item.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                ))
                            )
                        }
                        <div ref={inboxEndRef} />
                    </div>

                    {/* Text Input */}
                    <div className="border-t p-4 pb-8 md:pb-4 bg-[var(--color-surface)]" style={{ borderColor: 'var(--color-border-subtle)' }}>
                        <div className="flex items-center gap-3">
                            <label className="p-2.5 rounded-full cursor-pointer transition-colors hover:bg-[var(--color-surface-elevated)]" style={{
                                color: 'var(--color-text-secondary)'
                            }}>
                                <Upload size={20} />
                                <input type="file" accept="image/*,audio/*" className="hidden" onChange={handleFileUpload} />
                            </label>

                            <div className="flex-1 flex items-center bg-[var(--color-surface-elevated)] rounded-full px-4 py-2 border border-transparent focus-within:border-[var(--color-border)] transition-colors">
                                <input
                                    type="text"
                                    value={textInput}
                                    onChange={(e) => setTextInput(e.target.value)}
                                    onKeyDown={(e) => {
                                        if (e.key === 'Enter' && !e.shiftKey) {
                                            handleSendText();
                                        }
                                    }}
                                    placeholder="Message..."
                                    className="flex-1 bg-transparent border-none focus:ring-0 focus:outline-none text-sm placeholder:text-[var(--color-text-tertiary)]"
                                    style={{ color: 'var(--color-text-primary)' }}
                                />
                                {editingMessageId && (
                                    <button
                                        onClick={() => {
                                            setEditingMessageId(null);
                                            setTextInput('');
                                        }}
                                        className="ml-2 p-1 rounded-full hover:bg-black/10 dark:hover:bg-white/10 transition-colors"
                                        title="Cancel Edit"
                                    >
                                        <X size={16} className="text-red-500" />
                                    </button>
                                )}
                            </div>

                            {/* Send / Save Button */}
                            <button
                                onClick={handleSendText}
                                disabled={!textInput.trim() && !editingMessageId}
                                className={`p-2.5 rounded-full transition-all flex-shrink-0 ${textInput.trim()
                                    ? 'bg-[var(--color-primary)] text-white shadow-lg hover:shadow-xl hover:scale-105 active:scale-95'
                                    : 'bg-[var(--color-surface-elevated)] text-[var(--color-text-disabled)] cursor-not-allowed'
                                    }`}
                            >
                                {editingMessageId ? <Check size={20} /> : <Send size={20} />}
                            </button>
                        </div>
                    </div>
                </aside >

                {/* Center Panel - Voice Orb */}
                <div className="flex-1 flex flex-col items-center justify-center py-2 md:py-8 lg:py-12 order-1 lg:order-2 h-full">
                    <div className="mb-6 md:mb-8 text-[10px] font-bold tracking-[0.2em] text-blue-400/50 uppercase select-none border border-blue-500/10 bg-blue-500/5 px-3 py-1 rounded-full backdrop-blur-sm">
                        Public Beta
                    </div>
                    <button
                        className="p-0 m-0 border-none bg-transparent flex items-center justify-center relative appearance-none focus:outline-none transition-transform active:scale-[0.98]"
                        onClick={connected ? disconnect : connectToLiveAPI}
                        title={connected ? "Tap to disconnect" : "Tap to connect"}
                    >
                        <div className="w-56 h-56 md:w-80 md:h-80 lg:w-96 lg:h-96 rounded-full bg-[var(--color-surface)] shadow-[0_0_40px_-10px_rgba(0,0,0,0.5)] overflow-hidden border-4 border-white bg-black">
                            <Orb
                                agentState={
                                    mode === AppMode.LISTENING ? 'listening' :
                                        mode === AppMode.THINKING ? 'thinking' :
                                            mode === AppMode.SPEAKING ? 'talking' :
                                                null
                                }
                                volumeMode="manual"
                                manualInput={mode === AppMode.LISTENING ? Math.min(1, volume * 5) : 0}
                                manualOutput={mode === AppMode.SPEAKING ? Math.min(1, volume * 5) : 0}
                                colors={
                                    mode === AppMode.LISTENING ? ["#D9A26A", "#F2D8B3"] : // Listening: Warm Peach/Sand
                                        mode === AppMode.SPEAKING ? ["#9CA3AF", "#E5E7EB"] : // Talking: Silver/White
                                            mode === AppMode.THINKING ? ["#9CA3AF", "#E5E7EB"] : // Thinking: Silver/White (matches talking)
                                                ["#A0B9D1", "#CADCFC"] // Idle: Soft Blue
                                }
                            />
                        </div>
                    </button>

                    {/* Waveform Visualization */}
                    <div className="h-16 w-56 md:w-80 lg:w-80 mt-6 opacity-80">
                        <LiveWaveform
                            active={mode === AppMode.LISTENING}
                            processing={mode === AppMode.SPEAKING}
                            barColor={mode === AppMode.SPEAKING ? "#9CA3AF" : "#D9A26A"}
                            barWidth={4}
                            barGap={2}
                            height={40}
                            volume={volume}
                        />
                    </div>

                    {/* Status */}
                    <div
                        className="mt-4 md:mt-8 px-3 py-1.5 md:px-4 md:py-2 rounded-full border text-xs md:text-sm font-medium"
                        style={{
                            backgroundColor: 'var(--color-surface)',
                            borderColor: 'var(--color-border)',
                            color: 'var(--color-text-secondary)'
                        }}
                    >
                        {mode === AppMode.LISTENING && "Listening..."}
                        {mode === AppMode.THINKING && "Processing..."}
                        {mode === AppMode.SPEAKING && "Speaking..."}
                        {mode === AppMode.IDLE && "Tap to Connect"}
                    </div>

                    {/* Interaction Hint */}
                    <div className="mt-3 text-[10px] font-medium tracking-wide uppercase opacity-40 select-none animate-pulse" style={{ color: 'var(--color-text-tertiary)' }}>
                        {connected ? "Tap orb to disconnect" : "Tap orb to start"}
                    </div>
                </div >

            </main >

            {/* Camera Feed - Draggable & Snappable */}
            {showCameraPreview && (
                <div
                    onPointerDown={(e) => {
                        e.preventDefault();
                        e.stopPropagation();
                        const rect = e.currentTarget.getBoundingClientRect();
                        setDragStart({ x: e.clientX - rect.left, y: e.clientY - rect.top });
                        setIsDragging(true);
                    }}
                    className={`fixed z-[60] rounded-lg border overflow-hidden transition-all touch-none select-none ${isDragging ? 'duration-0 scale-105 shadow-xl cursor-grabbing' : 'duration-500 ease-out cursor-grab'} top-24 right-3 md:top-auto md:bottom-24 md:right-6`}
                    style={{
                        backgroundColor: 'var(--color-bg)',
                        borderColor: 'var(--color-border)',
                        left: cameraPos ? cameraPos.x : undefined,
                        top: cameraPos ? cameraPos.y : undefined,
                        right: cameraPos ? 'auto' : undefined,
                        bottom: cameraPos ? 'auto' : undefined,
                        touchAction: 'none'
                    }}
                >
                    {/* Global Drag Listeners */}
                    {isDragging && (
                        null
                    )}

                    {/* We need to apply size classes conditionally or via style to ensure drag logic knows size */}
                    <div className={`w-20 aspect-[9/16] md:w-32 lg:w-48 md:aspect-video pointer-events-none`} />
                    <video
                        ref={videoRef}
                        autoPlay
                        playsInline
                        muted
                        className="absolute inset-0 w-full h-full object-cover pointer-events-none"
                        style={{ transform: 'scaleX(-1)' }}
                    />
                    {
                        !cameraActive && (
                            <div className="absolute inset-0 flex items-center justify-center text-xs pointer-events-none" style={{
                                backgroundColor: 'var(--color-surface)',
                                color: 'var(--color-text-tertiary)'
                            }}>
                                Camera Off
                            </div>
                        )
                    }
                    {
                        cameraActive && (
                            <div
                                className="absolute top-2 left-2 px-2 py-1 rounded-full text-xs font-medium flex items-center gap-1.5 pointer-events-none"
                                style={{ backgroundColor: 'var(--color-error)', color: 'white' }}
                            >
                                <div className="w-1 h-1 rounded-full bg-white animate-pulse" />
                                Live
                            </div>
                        )
                    }
                </div >
            )}

            {/* Hidden Canvas */}
            < canvas ref={canvasRef} className="hidden" />





            {/* Debug Logs - Desktop Only */}
            <div className="hidden lg:block fixed top-20 right-4 w-64 max-h-48 overflow-y-auto text-xs opacity-20 hover:opacity-100 transition-opacity pointer-events-none hover:pointer-events-auto z-30">
                <div className="space-y-1">
                    {logs.slice(-5).map((log, i) => (
                        <div
                            key={i}
                            className="px-2 py-1 rounded font-mono"
                            style={{ backgroundColor: 'var(--color-surface)' }}
                        >
                            <span style={{
                                color: log.role === 'user' ? 'var(--color-success)' :
                                    log.role === 'assistant' ? 'var(--color-accent)' :
                                        'var(--color-text-tertiary)'
                            }}>
                                [{log.role}]
                            </span>{' '}
                            <span style={{ color: 'var(--color-text-secondary)' }}>{log.text}</span>
                        </div>
                    ))}
                </div>
            </div>

            {showWelcome && <WelcomeScreen onComplete={handleWelcomeComplete} />}

        </div >
    );
};

export default App;
