import os
import io
from typing import List, Tuple, Optional
import pyaudio
import wave
from openai import OpenAI
import asyncio
from threading import Thread, Event
import webrtcvad
from utils import transcription_concat
import tempfile
from text_filter import TextFilterManager, FilterError
from mode_detector import ModeDetector, ModeDetectionError
from rich import box
from rich.align import Align
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text
from rich.panel import Panel
from rich.layout import Layout
from collections import deque
from datetime import datetime

FORMAT = pyaudio.paInt16  # Audio format
CHANNELS = 1  # Mono audio
RATE = 16000  # Sample rate
CHUNK_DURATION_MS = 30  # Frame duration in milliseconds
CHUNK = int(RATE * CHUNK_DURATION_MS / 1000)
MIN_SPEECH_RATIO = 0
MIN_TRANSCRIPTION_SIZE_MS = int(
    os.getenv('UTTERTYPE_MIN_TRANSCRIPTION_SIZE_MS', 1000) # Minimum duration of speech to send to API in case of silence
)

class LogFormatter:
    """Handles consistent formatting of log messages across the application."""
    
    # Log types and their formatting rules
    LOG_TYPES = {
        'INFO': {'style': 'blue', 'prefix': '🔵'},
        'WARN': {'style': 'yellow', 'prefix': '⚠️'},
        'ERROR': {'style': 'red', 'prefix': '❌'},
        'SUCCESS': {'style': 'green', 'prefix': '✅'},
        'SPEECH': {'style': 'cyan', 'prefix': '🎤'},
        'TTS': {'style': 'magenta', 'prefix': '🔊'}  # For future TTS functionality
    }
    
    @classmethod
    def format(cls, message: str, log_type: str = 'INFO', should_tts: bool = False) -> Text:
        """Format a log message with consistent styling.
        
        Args:
            message: The message to format
            log_type: The type of log message (INFO, WARN, ERROR, SUCCESS, SPEECH)
            should_tts: Whether this message should be sent to TTS (future functionality)
        
        Returns:
            Rich Text object with formatted message
        """
        log_format = cls.LOG_TYPES.get(log_type, cls.LOG_TYPES['INFO'])
        timestamp = datetime.now().strftime('[%H:%M:%S]')
        
        text = Text()
        text.append(timestamp, style='dim')
        text.append(' ')
        text.append(log_format['prefix'], style=log_format['style'])
        text.append(' ')
        text.append(message, style=log_format['style'])
        
        if should_tts:
            text.append(' 🔊', style='magenta')
            
        return text


class TranscriberLogger:
    """Handles all logging functionality for the AudioTranscriber"""
    
    # Define states and their associated styles
    STATES = {
        'IDLE': {'color': 'dim', 'symbol': '⚪', 'label': 'Idle'},
        'SILENCE': {'color': 'red', 'symbol': '🔴', 'label': 'Silence'},
        'SPEECH': {'color': 'green', 'symbol': '🟢', 'label': 'Speech'},
        'PROCESSING': {'color': 'yellow', 'symbol': '🟡', 'label': 'Processing'}
    }
    
    def __init__(self):
        self.console = Console()
        self.console_table = None
        
    def set_console_table(self, console_table):
        """Set the console table to use for logging"""
        self.console_table = console_table
        
    def log_message(self, message: str, log_type: str = 'INFO', should_tts: bool = False):
        """Log a message with consistent formatting.
        
        Args:
            message: The message to log
            log_type: Type of log message (INFO, WARN, ERROR, SUCCESS, SPEECH)
            should_tts: Whether this message should be sent to TTS (future)
        """
        formatted_message = LogFormatter.format(message, log_type, should_tts)
        if self.console_table:
            self.console_table.add_log(formatted_message)
        else:
            self.console.print(formatted_message)
            
    def _create_status_text(self, label: str, value: str, style: str = None) -> Text:
        """Create a styled status text component with consistent formatting."""
        text = Text()
        text.append(f"\n{label}: ", style="bold" if style == "header" else "dim")
        text.append(value, style=style if style else "default")
        return text
    
    def _format_count(self, count: int, warning_threshold: int = 0) -> Text:
        """Format a count with appropriate styling based on threshold."""
        style = "yellow bold" if count > warning_threshold else "dim"
        return Text(str(count), style=style)
    
    def update_vad_status(self, is_speech: bool, current_audio_duration: int, stats: dict):
        """Update the VAD status indicator with current state and statistics."""
        # Determine current state
        if stats['active_requests'] > 0:
            current_state = self.STATES['PROCESSING']
        elif is_speech:
            current_state = self.STATES['SPEECH']
        else:
            current_state = self.STATES['SILENCE']
            
        # Create main status indicator
        status = Text()
        status.append(current_state['symbol'], style=f"{current_state['color']} bold")
        status.append(f" {current_state['label']}", style=current_state['color'])
        status.append(f" ({current_audio_duration}ms)", style="dim")
        
        # Add buffer information
        buffer_size = stats['buffer_size']
        buffer_percentage = min(100, int((buffer_size / MIN_TRANSCRIPTION_SIZE_MS) * 100))
        buffer_style = "green bold" if buffer_percentage >= 100 else "yellow"
        status.append(self._create_status_text(
            "Buffer",
            f"{buffer_size}ms ({buffer_percentage}%)",
            buffer_style
        ))
        
        # Add statistics section
        status.append(Text("\nStatistics:", style="bold"))
        
        # Speech/Silence ratio
        ratio_text = Text("\nSpeech/Silence: ", style="dim")
        ratio_text.append(Text(str(stats['speech_frames_count']), style="green"))
        ratio_text.append(Text("/", style="dim"))
        ratio_text.append(Text(str(stats['silence_frames_count']), style="red"))
        status.append(ratio_text)
        
        # Add remaining stats in a cleaner format
        stats_to_show = {
            "Transcriptions": ("transcription_count", "blue bold"),
            "Discarded": ("discarded_buffers", None),
            "Active Requests": ("active_requests", None),
            "Last Action": ("last_action", "magenta")
        }
        
        for label, (key, style) in stats_to_show.items():
            status.append(self._create_status_text(label, str(stats[key]), style))
        
        # Display the status
        if self.console_table:
            self.console_table.update_vad_status(status)
        else:
            self.console.print(status)
            
        return "Met" if buffer_percentage >= 100 else "Not Met", status


class AudioTranscriber:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.recording_finished = Event()  # Threading event to end recording
        self.recording_finished.set()  # Initialize as finished
        self.frames = []
        self.audio_duration = 0
        self.rolling_transcriptions: List[Tuple[int, str]] = []  # (idx, transcription)
        self.rolling_requests: List[Thread] = []  # list of pending requests
        self.event_loop = asyncio.get_event_loop()
        self.vad = webrtcvad.Vad(1)  # Most aggressive filtering (0-3 scale)
        self.transcriptions = asyncio.Queue()
        self.filter_manager = TextFilterManager()  # Initialize text filter manager
        self.mode_detector = ModeDetector()  # Initialize mode detector for voice commands
        
        # Create logger
        self.logger = TranscriberLogger()
        
        # Stats and counters
        self.speech_frames_count = 0
        self.silence_frames_count = 0
        self.total_frames_processed = 0
        self.transcription_count = 0
        self.active_requests = 0
        self.buffer_size = 0
        self.discarded_buffers = 0  # Counter for discarded silence buffers
        self.last_action = "Initialized"
        self.threshold_status = "N/A"
        self.vad_status = None  # Current VAD status indicator
        
    def set_logger(self, console_table):
        """Set the logger to use for logging messages"""
        self.logger.set_console_table(console_table)
        
    def log_message(self, message: str, log_type: str = 'INFO', should_tts: bool = False):
        """Log a message using the logger with optional type and TTS flag.
        
        Args:
            message: The message to log
            log_type: Type of log message (INFO, WARN, ERROR, SUCCESS, SPEECH)
            should_tts: Whether this message should be sent to TTS (future)
        """
        self.logger.log_message(message, log_type=log_type, should_tts=should_tts)
        
    def get_stats(self):
        """Get current statistics for logging purposes"""
        buffer_size = len(self.frames) * CHUNK_DURATION_MS
        return {
            'speech_frames_count': self.speech_frames_count,
            'silence_frames_count': self.silence_frames_count,
            'total_frames_processed': self.total_frames_processed,
            'transcription_count': self.transcription_count,
            'active_requests': self.active_requests,
            'buffer_size': buffer_size,
            'discarded_buffers': self.discarded_buffers,
            'last_action': self.last_action,
            'threshold_status': self.threshold_status
        }
        
    def update_vad_indicator(self, is_speech, current_audio_duration):
        """Update the VAD status indicator via the logger"""
        stats = self.get_stats()
        self.threshold_status, self.vad_status = self.logger.update_vad_status(
            is_speech, 
            current_audio_duration, 
            stats
        )
    
    def start_recording(self):
        """Start recording audio from the microphone."""
        def _record():
            self.recording_finished = Event()
            stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
            )
            
            self.log_message("Started recording", log_type='INFO', should_tts=True)
            intermediate_trancriptions_idx = 0
            
            while not self.recording_finished.is_set():
                data = stream.read(CHUNK)
                self.audio_duration += CHUNK_DURATION_MS
                is_speech = self.vad.is_speech(data, RATE)
                current_audio_duration = len(self.frames) * CHUNK_DURATION_MS
                
                # Update frame statistics
                self.total_frames_processed += 1
                if is_speech:
                    self.speech_frames_count += 1
                    self.last_action = "Adding speech to buffer"
                else:
                    self.silence_frames_count += 1
                    self.last_action = "Detected silence"
                
                if self.audio_duration % 30 == 0:
                    self.update_vad_indicator(is_speech, current_audio_duration)
                
                if (not is_speech and current_audio_duration >= MIN_TRANSCRIPTION_SIZE_MS):
                    frames_in_buffer = len(self.frames)
                    min_speech_ratio = MIN_SPEECH_RATIO
                    speech_ratio = self.speech_frames_count / max(1, frames_in_buffer)
                    
                    if speech_ratio < min_speech_ratio:
                        if self.total_frames_processed % 100 == 0:
                            self.log_message(
                                f"Low speech content in buffer (ratio: {speech_ratio:.2f}, min: {min_speech_ratio})",
                                log_type='WARN'
                            )
                        self.discarded_buffers += 1
                        self.frames = []
                        continue
                        
                    self.last_action = "Processing speech segment"
                    self.log_message(
                        f"Speech segment detected ({current_audio_duration}ms)",
                        log_type='SPEECH'
                    )
                    self.update_vad_indicator(is_speech, current_audio_duration)
                    
                    rolling_request = Thread(
                        target=self._intermediate_transcription,
                        args=(intermediate_trancriptions_idx, self._frames_to_wav()),
                    )
                    self.frames = []
                    self.active_requests += 1
                    self.rolling_requests.append(rolling_request)
                    rolling_request.start()
                    intermediate_trancriptions_idx += 1
                    self.transcription_count += 1
                    
                    self.update_vad_indicator(is_speech, 0)
                self.frames.append(data)

        Thread(target=_record).start()

    def stop_recording(self):
        """Stop the recording and reset variables"""
        self.last_action = "Stopping recording"
        self.log_message("Stopping recording", log_type='INFO', should_tts=True)
        self.update_vad_indicator(False, len(self.frames) * CHUNK_DURATION_MS)
        
        if len(self.frames) > 0:
            self.log_message(
                f"Processing final audio segment ({len(self.frames) * CHUNK_DURATION_MS}ms)",
                log_type='SPEECH'
            )
            self.transcription_count += 1
            
        self.recording_finished.set()
        self._finish_transcription()
        
        # Reset all variables
        self.frames = []
        self.audio_duration = 0
        self.rolling_requests = []
        self.rolling_transcriptions = []
        self.speech_frames_count = 0
        self.silence_frames_count = 0
        self.total_frames_processed = 0
        self.active_requests = 0
        self.transcription_count = 0
        self.discarded_buffers = 0
        self.threshold_status = "N/A"
        self.last_action = "Recording stopped"
        self.update_vad_indicator(False, 0)

    def _intermediate_transcription(self, idx, audio):
        self.last_action = f"Processing transcription #{idx}"
        self.log_message(f"Processing transcription #{idx}", log_type='INFO')
        self.update_vad_indicator(False, len(self.frames) * CHUNK_DURATION_MS)
        
        intermediate_transcription = self.transcribe_audio(audio)
        self.rolling_transcriptions.append((idx, intermediate_transcription))
        self.active_requests -= 1
        self.last_action = f"Completed transcription #{idx}"
        self.log_message(
            f"Completed transcription #{idx}: '{intermediate_transcription[:30]}...'",
            log_type='SUCCESS'
        )
        self.update_vad_indicator(False, len(self.frames) * CHUNK_DURATION_MS)

    async def _apply_text_filter(self, text: str, filter_name: Optional[str] = None) -> str:
        """
        Apply text filtering to the transcription.
        
        Args:
            text: Text to filter
            filter_name: Optional name of specific filter to use
            
        Returns:
            Filtered text
            
        Note: If filtering fails, logs the error and returns the original text
        """
        try:
            # Check if text contains a mode switching command
            try:
                detected_mode = await self.mode_detector.detect_mode_switch(text)
                if detected_mode:
                    # Switch mode and log the change
                    self.log_message(f"Voice command detected: Switching to '{detected_mode}' mode")
                    self.filter_manager.set_filter(detected_mode)
                    # Return empty string instead of mode switch confirmation message
                    return ""
            except ModeDetectionError as e:
                # Log error but continue with normal text processing
                self.log_message(f"Mode detection error: {str(e)}")
            
            # Process text with current or specified filter
            return await self.filter_manager.process_text(text, filter_name)
        except FilterError as e:
            # Log error but continue with unfiltered text
            self.log_message(f"Text filtering error: {str(e)}")
            return text

    def _finish_transcription(self):
        self.last_action = "Finalizing transcription"
        self.update_vad_indicator(False, len(self.frames) * CHUNK_DURATION_MS)
        
        # Check if there's enough speech content to transcribe
        if len(self.frames) > 0:
            min_speech_ratio = float(os.getenv('UTTERTYPE_MIN_SPEECH_RATIO', "0.1"))
            speech_ratio = self.speech_frames_count / max(1, len(self.frames))
            
            if speech_ratio < min_speech_ratio:
                self.log_message(
                    f"No speech detected in buffer (ratio: {speech_ratio:.2f}, threshold: {min_speech_ratio})",
                    log_type='WARN'
                )
                self.discarded_buffers += 1
                # Still process any pending transcriptions
                for request in self.rolling_requests:
                    request.join()
                    
                if self.rolling_transcriptions:
                    # Process only the existing transcriptions if we have any
                    sorted(self.rolling_transcriptions, key=lambda x: x[0])
                    transcriptions = [t[1] for t in self.rolling_transcriptions]
                    combined_transcription = transcription_concat(transcriptions)
                    
                    self.event_loop.call_soon_threadsafe(
                        self.transcriptions.put_nowait,
                        (combined_transcription, self.audio_duration),
                    )
                return
            
        transcription = self.transcribe_audio(self._frames_to_wav())
        for request in self.rolling_requests:
            request.join()
            
        self.rolling_transcriptions.append(
            (len(self.rolling_transcriptions), transcription)
        )
        sorted(self.rolling_transcriptions, key=lambda x: x[0])
        transcriptions = [t[1] for t in self.rolling_transcriptions]
        combined_transcription = transcription_concat(transcriptions)
        
        self.event_loop.call_soon_threadsafe(
            self.transcriptions.put_nowait,
            (combined_transcription, self.audio_duration),
        )
        self.last_action = "Transcription finalized"
        self.log_message("Transcription completed and queued", log_type='SUCCESS')

    def _frames_to_wav(self):
        buffer = io.BytesIO()
        buffer.name = "tmp.wav"
        wf = wave.open(buffer, "wb")
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b"".join(self.frames))
        wf.close()
        return buffer

    def transcribe_audio(self, audio: io.BytesIO) -> str:
        raise NotImplementedError("Please use a subclass of AudioTranscriber")

    async def get_transcriptions(self):
        """
        Asynchronously get transcriptions from the queue.
        Returns (raw transcription, filtered transcription, current mode, audio duration in ms).
        
        Applies text filtering before yielding the result.
        """
        while True:
            raw_transcription, audio_duration = await self.transcriptions.get()
            
            # Get the current filter mode name
            current_mode = self.filter_manager.current_filter
            
            # Apply the current text filter to the transcription
            filtered_transcription = await self._apply_text_filter(raw_transcription)
            
            yield raw_transcription, filtered_transcription, current_mode, audio_duration
            self.transcriptions.task_done()
            
    def get_available_filters(self):
        """
        Get all available text filter modes.
        
        Returns:
            Dictionary of filter names and descriptions
        """
        return self.filter_manager.get_available_filters()
    
    def get_current_filter(self):
        """
        Get the name of the current text filter mode.
        
        Returns:
            Name of the current filter
        """
        return self.filter_manager.current_filter
    
    def set_filter(self, filter_name: str):
        """
        Set the current text filter mode.
        
        Args:
            filter_name: Name of the filter to set
            
        Raises:
            ValueError: If the filter doesn't exist
        """
        self.filter_manager.set_filter(filter_name)


class WhisperAPITranscriber(AudioTranscriber):
    def __init__(self, base_url, model_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = model_name
        self.client = OpenAI(base_url=base_url)
        self.log_message(f"Initialized Whisper API transcriber with model: {model_name}", log_type='INFO')

    @staticmethod
    def create(*args, **kwargs):
        base_url = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
        model_name = os.getenv('OPENAI_MODEL_NAME', 'whisper-1')
        return WhisperAPITranscriber(base_url, model_name)

    def transcribe_audio(self, audio: io.BytesIO) -> str:
        try:
            transcription = self.client.audio.transcriptions.create(
                model=self.model_name,
                file=audio,
                response_format="text",
                language="en",
                prompt=os.getenv('WHISPER_PROMPT', ""),
                extra_body={
                    "vad_filter": True,
                    "temperature": float(os.getenv('WHISPER_TEMPERATURE', "0")),
                    "verbose": os.getenv('WHISPER_VERBOSE', "true").lower() == "true",
                    "task": os.getenv('WHISPER_TASK', "transcribe"),
                    "suppress_tokens": os.getenv('WHISPER_SUPPRESS_TOKENS', ""),
                    "condition_on_previous_text": os.getenv('WHISPER_CONDITION_ON_PREVIOUS_TEXT', "true").lower() == "true",
                }
            )
            return transcription
        except Exception as e:
            self.log_message(f"Transcription error: {str(e)}", log_type='ERROR')
            return ""


class WhisperLocalMLXTranscriber(AudioTranscriber):
    def __init__(self, model_type="distil-medium.en", *args, **kwargs):
        super().__init__(*args, **kwargs)
        from lightning_whisper_mlx import LightningWhisperMLX
        self.model = LightningWhisperMLX(model_type)
        self.log_message(f"Initialized local MLX transcriber with model: {model_type}", log_type='INFO')

    def transcribe_audio(self, audio: io.BytesIO) -> str:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
                tmpfile.write(audio.getvalue())
                transcription = self.model.transcribe(tmpfile.name)["text"]
                os.unlink(tmpfile.name)
            return transcription
        except Exception as e:
            self.log_message(f"Transcription error: {str(e)}", log_type='ERROR')
            return ""
