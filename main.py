import asyncio
import logging
from pynput import keyboard
from transcriber import WhisperAPITranscriber
from table_interface import ConsoleTable
from keyboard_listener import create_keylistener
from dotenv import load_dotenv
from utils import manual_type


async def main():
    load_dotenv()
    
    # Suppress standard logging to console before UI is initialized
    root_logger = logging.getLogger()
    root_logger.handlers = []  # Remove any existing handlers
    root_logger.addHandler(logging.NullHandler())  # Add null handler to suppress output
    
    # Initialize console table first
    console_table = ConsoleTable()
    
    # Now initialize transcriber and components
    transcriber = WhisperAPITranscriber.create()
    hotkey = create_keylistener(transcriber)

    keyboard.Listener(on_press=hotkey.press, on_release=hotkey.release).start()
    
    # Connect the logger to the transcriber and its components
    transcriber.set_logger(console_table)
    transcriber.filter_manager.set_logger(console_table.add_log)
    transcriber.mode_detector.set_logger(console_table.add_log)
    
    with console_table:
        # Log available filters to the console table
        console_table.add_log("Starting uttertype with the following filters:")
        for name, description in transcriber.get_available_filters().items():
            console_table.add_log(f"Filter: {name} - {description}")
        console_table.add_log(f"Current filter: {transcriber.get_current_filter()}")
        
        async for raw_transcription, filtered_transcription, current_mode, audio_duration_ms in transcriber.get_transcriptions():
            # Type only the filtered text
            manual_type(filtered_transcription.strip())
            # Log both raw and filtered text to the console table
            console_table.insert(
                raw_transcription,
                current_mode,
                filtered_transcription,
                round(0.0001 * audio_duration_ms / 1000, 6),
            )


if __name__ == "__main__":
    asyncio.run(main())
