import os
import sys
from pynput.keyboard import HotKey, Key, KeyCode
from pynput.keyboard._darwin import KeyCode as DarwinKeyCode


class HoldHotKey(HotKey):
    """A HotKey that supports both activation and deactivation callbacks."""
    
    def __init__(self, keys, on_activate, on_deactivate):
        self.active = False

        def _mod_on_activate():
            self.active = True
            on_activate()

        def _mod_on_deactivate():
            self.active = False
            on_deactivate()

        super().__init__(keys, _mod_on_activate)
        self._on_deactivate = _mod_on_deactivate

    def release(self, key):
        super().release(key)
        if self.active and self._state != self._keys:
            self._on_deactivate()


class CtrlGlobeHotKey:
    """
    macOS-specific handler for Ctrl+Globe key combination.
    Handles the unique behavior of the Globe key on macOS, which only sends release events.
    """
    def __init__(self, on_activate, on_deactivate):
        self.current_keys = set()
        self._on_activate = on_activate
        self._on_deactivate = on_deactivate
        self.active = False

    def press(self, key):
        if key in [Key.ctrl, Key.ctrl_l, Key.ctrl_r]:
            self.current_keys.add('ctrl')
            self._check_activation()

    def release(self, key):
        # Handle Globe key toggle (comes through as release event)
        if isinstance(key, DarwinKeyCode) and hasattr(key, "vk") and key.vk == 63:
            if 'globe' in self.current_keys:
                self.current_keys.discard('globe')
            else:
                self.current_keys.add('globe')
            self._check_activation()
        
        # Handle control key release
        elif key in [Key.ctrl, Key.ctrl_l, Key.ctrl_r]:
            self.current_keys.discard('ctrl')
            if self.active:
                self.active = False
                self._on_deactivate()
    
    def _check_activation(self):
        """Check if we should activate or deactivate based on current key state."""
        if 'ctrl' in self.current_keys and 'globe' in self.current_keys and not self.active:
            self.active = True
            self._on_activate()
        elif self.active and not ('ctrl' in self.current_keys and 'globe' in self.current_keys):
            self.active = False
            self._on_deactivate()


class HoldGlobeKey:
    """
    macOS-specific handler for Globe key.
    Handles the unique toggle behavior of the Globe key on macOS.
    """
    def __init__(self, on_activate, on_deactivate):
        self.held = False
        self._on_activate = on_activate
        self._on_deactivate = on_deactivate

    def press(self, key):
        if hasattr(key, "vk") and key.vk == 63:
            if self.held:
                self._on_deactivate()
            else:
                self._on_activate()
            self.held = not self.held

    def release(self, key):
        """Press and release signals are mixed for globe key"""
        self.press(key)


def create_keylistener(transcriber, env_var="UTTERTYPE_RECORD_HOTKEYS"):
    """
    Create an appropriate key listener based on environment settings and platform.
    
    Args:
        transcriber: Object with start_recording and stop_recording methods
        env_var: Environment variable name for hotkey configuration
        
    Returns:
        A key listener appropriate for the platform and configuration
    """
    key_code = os.getenv(env_var, "")

    # Handle Ctrl+Globe combination for macOS
    if sys.platform == "darwin" and key_code in ["<ctrl>+<globe>", "ctrl+globe"]:
        return CtrlGlobeHotKey(
            on_activate=transcriber.start_recording,
            on_deactivate=transcriber.stop_recording,
        )
    # Handle Globe key alone for macOS
    elif (sys.platform == "darwin") and (key_code in ["<globe>", ""]):
        return HoldGlobeKey(
            on_activate=transcriber.start_recording,
            on_deactivate=transcriber.stop_recording,
        )

    # Default to standard hotkey combinations
    key_code = key_code if key_code else "<ctrl>+<alt>+v"
    return HoldHotKey(
        HoldHotKey.parse(key_code),
        on_activate=transcriber.start_recording,
        on_deactivate=transcriber.stop_recording,
    )