"""
TensorFlow-based audio processor with robust VAD and noise suppression.
If TensorFlow is not available in the environment, a lightweight NumPy
fallback is used so the application can start without heavy ML deps.
"""
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception:
    tf = None
    TF_AVAILABLE = False

import numpy as np
from collections import deque
from typing import Tuple


class TFAudioProcessor:
    """
    Handles audio normalization, noise suppression, and robust VAD.
    - Noise Suppression: Spectral subtraction for background noise reduction
    - Normalization: Dynamic gain control for consistent volume
    - Conversion: Float32 ↔ Int16 PCM
    - VAD: Multi-feature detection (energy + ZCR + spectral) with adaptive noise floor
    """
    
    def __init__(
        self, 
        target_sample_rate: int = 16000,  # Gemini uses 16kHz for input
        vad_energy_threshold: float = 0.025,  # Raised from 0.02 for stricter detection
        vad_zcr_threshold: float = 0.15,  # Zero-crossing rate threshold
        min_gain: float = 0.5,
        max_gain: float = 3.0,
        noise_alpha: float = 0.95  # Noise floor smoothing factor
    ):
        self.target_sample_rate = target_sample_rate
        self.vad_energy_threshold = vad_energy_threshold
        self.vad_zcr_threshold = vad_zcr_threshold
        self.min_gain = min_gain
        self.max_gain = max_gain
        self.noise_alpha = noise_alpha
        
        # State for robust VAD with hysteresis
        self._energy_history = deque(maxlen=150)  # Longer history for better noise estimation
        self._zcr_history = deque(maxlen=50)
        self._noise_floor_energy = 0.01
        self._noise_floor_zcr = 0.1
        self._is_speaking = False
        self._speech_frames = 0
        self._silence_frames = 0

        # Stricter hysteresis thresholds (frames) to avoid false positives
        self._speech_threshold = 5   # Frames of speech to confirm speaking (increased from 3)
        self._silence_threshold = 12  # Frames of silence to confirm stopped (increased from 10)
        
        # Noise suppression state
        self._noise_spectrum = None
        self._frame_count = 0

        # Voice analytics state (for confidence estimation)
        self._pitch_history = deque(maxlen=100)
        self._pitch_jitter_history = deque(maxlen=100)
        self._last_pitch = 0.0
        self._total_frames = 0
        self._voiced_frames = 0
        self._speech_segment_count = 0
        self._prev_frame_was_speech = False
    
    def _spectral_noise_suppression(self, audio_tensor):
        """
        Apply spectral noise suppression. Works with TensorFlow tensors when
        available, otherwise accepts a NumPy array and uses a simple spectral
        subtraction fallback.
        """
        if TF_AVAILABLE and isinstance(audio_tensor, (tf.Tensor,)):
            # original TF implementation
            fft = tf.signal.rfft(audio_tensor)
            magnitude = tf.abs(fft)
            phase = tf.math.angle(fft)

            if self._noise_spectrum is None or self._frame_count < 50:
                if self._noise_spectrum is None:
                    self._noise_spectrum = magnitude.numpy()
                else:
                    self._noise_spectrum = (
                        self.noise_alpha * self._noise_spectrum +
                        (1 - self.noise_alpha) * magnitude.numpy()
                    )
                self._frame_count += 1

            noise_tensor = tf.constant(self._noise_spectrum, dtype=tf.float32)
            suppressed_magnitude = tf.maximum(magnitude - 1.5 * noise_tensor, magnitude * 0.1)
            suppressed_fft = tf.cast(suppressed_magnitude, tf.complex64) * tf.exp(tf.complex(0.0, phase))
            suppressed_audio = tf.signal.irfft(suppressed_fft)
            original_length = tf.shape(audio_tensor)[0]
            return suppressed_audio[:original_length]

        # NumPy fallback: simple spectral subtraction using numpy FFT
        audio_np = np.asarray(audio_tensor, dtype=np.float32)
        fft = np.fft.rfft(audio_np)
        magnitude = np.abs(fft)
        phase = np.angle(fft)
        if self._noise_spectrum is None or self._frame_count < 50:
            if self._noise_spectrum is None:
                self._noise_spectrum = magnitude
            else:
                self._noise_spectrum = self.noise_alpha * self._noise_spectrum + (1 - self.noise_alpha) * magnitude
            self._frame_count += 1
        suppressed_magnitude = np.maximum(magnitude - 1.5 * self._noise_spectrum, magnitude * 0.1)
        suppressed_fft = suppressed_magnitude * np.exp(1j * phase)
        suppressed_audio = np.fft.irfft(suppressed_fft)
        return suppressed_audio[: len(audio_np)]
    
    def _compute_zero_crossing_rate(self, audio_tensor) -> float:
        """
        Compute zero-crossing rate (ZCR) - indicator of speech vs noise.
        Speech typically has moderate ZCR, noise has high or low ZCR.
        
        Args:
            audio_tensor: Float32 audio tensor
            
        Returns:
            Zero-crossing rate (0.0 to 1.0)
        """
        if TF_AVAILABLE and isinstance(audio_tensor, (tf.Tensor,)):
            signs = tf.sign(audio_tensor)
            sign_changes = tf.abs(signs[1:] - signs[:-1])
            zcr = tf.reduce_sum(sign_changes) / (2.0 * tf.cast(tf.size(audio_tensor), tf.float32))
            return float(zcr)
        arr = np.asarray(audio_tensor)
        signs = np.sign(arr)
        sign_changes = np.abs(signs[1:] - signs[:-1])
        zcr = np.sum(sign_changes) / (2.0 * float(arr.size))
        return float(zcr)
    
    def _compute_spectral_centroid(self, audio_tensor) -> float:
        """
        Compute spectral centroid - indicator of brightness/frequency content.
        Speech has characteristic centroid range, noise often different.
        
        Args:
            audio_tensor: Float32 audio tensor
            
        Returns:
            Normalized spectral centroid
        """
        if TF_AVAILABLE and isinstance(audio_tensor, (tf.Tensor,)):
            fft = tf.signal.rfft(audio_tensor)
            magnitude = tf.abs(fft)
            freqs = tf.range(0, tf.shape(magnitude)[0], dtype=tf.float32)
            centroid = tf.reduce_sum(freqs * magnitude) / (tf.reduce_sum(magnitude) + 1e-8)
            normalized_centroid = centroid / (tf.cast(tf.shape(magnitude)[0], tf.float32))
            return float(normalized_centroid)
        arr = np.asarray(audio_tensor)
        fft = np.fft.rfft(arr)
        magnitude = np.abs(fft)
        freqs = np.arange(0, magnitude.shape[0], dtype=np.float32)
        centroid = np.sum(freqs * magnitude) / (np.sum(magnitude) + 1e-8)
        normalized_centroid = centroid / float(magnitude.shape[0])
        return float(normalized_centroid)

    def _estimate_pitch_hz(self, audio_array: np.ndarray) -> float:
        """
        Estimate fundamental frequency (pitch) using autocorrelation.
        Returns 0.0 if pitch is unreliable.
        """
        if audio_array.size < 2:
            return 0.0

        # Simple energy gate to avoid estimating pitch on silence
        energy = float(np.sqrt(np.mean(np.square(audio_array))))
        if energy < 1e-3:
            return 0.0

        # Autocorrelation
        audio = audio_array - np.mean(audio_array)
        corr = np.correlate(audio, audio, mode='full')
        corr = corr[corr.size // 2 :]  # keep non-negative lags

        # Limit search to plausible human pitch range (70–300 Hz)
        min_lag = int(self.target_sample_rate / 300)
        max_lag = int(self.target_sample_rate / 70)
        if max_lag >= corr.size:
            max_lag = corr.size - 1
        if min_lag >= max_lag:
            return 0.0

        segment = corr[min_lag:max_lag]
        peak_index = np.argmax(segment) + min_lag
        if corr[peak_index] <= 0:
            return 0.0

        pitch_hz = float(self.target_sample_rate / peak_index)
        return pitch_hz if 50.0 <= pitch_hz <= 400.0 else 0.0
    
    def process_audio(self, audio_data: bytes, input_format: str = 'float32') -> Tuple[bytes, bool, float]:
        """
        Process raw audio bytes with noise suppression, normalization, and robust VAD.
        
        Args:
            audio_data: Raw audio bytes
            input_format: 'float32' or 'int16'
            
        Returns:
            Tuple of (processed PCM16 bytes, is_speech boolean, confidence score 0.0-1.0)
        """
        # 1. Convert bytes to Float32 array
        if input_format == 'float32':
            audio_array = np.frombuffer(audio_data, dtype=np.float32)
        elif input_format == 'int16':
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            audio_array = np.frombuffer(audio_data, dtype=np.float32)

        if len(audio_array) == 0:
            return b'', False, 0.0

        # Keep a NumPy view for analytics regardless of TF availability
        audio_np = None

        if TF_AVAILABLE:
            tensor = tf.constant(audio_array, dtype=tf.float32)
            try:
                tensor = self._spectral_noise_suppression(tensor)
            except Exception:
                pass
            peak = float(tf.reduce_max(tf.abs(tensor)))
            if peak > 0.001:
                target_gain = 0.7 / peak
                gain = np.clip(target_gain, self.min_gain, self.max_gain)
                tensor = tensor * gain
                tensor = tf.clip_by_value(tensor, -1.0, 1.0)

            audio_np = tensor.numpy()
            energy = float(tf.sqrt(tf.reduce_mean(tf.square(tensor))))
            self._energy_history.append(energy)
            zcr = self._compute_zero_crossing_rate(tensor)
            self._zcr_history.append(zcr)
            spectral_centroid = self._compute_spectral_centroid(tensor)
        else:
            # NumPy fallback processing (simpler operations)
            audio = audio_array.astype(np.float32)
            # Simple DC removal / noise suppression: subtract median
            audio = audio - np.median(audio)
            # Normalize
            peak = float(np.max(np.abs(audio)))
            if peak > 0.001:
                target_gain = 0.7 / peak
                gain = np.clip(target_gain, self.min_gain, self.max_gain)
                audio = np.clip(audio * gain, -1.0, 1.0)

            audio_np = audio
            energy = float(np.sqrt(np.mean(np.square(audio))))
            self._energy_history.append(energy)
            zcr = self._compute_zero_crossing_rate(audio)
            self._zcr_history.append(zcr)
            spectral_centroid = self._compute_spectral_centroid(audio)

        # Update adaptive noise floors
        if len(self._energy_history) >= 30:
            sorted_energy = sorted(self._energy_history)
            self._noise_floor_energy = sorted_energy[len(sorted_energy) * 30 // 100]

        if len(self._zcr_history) >= 20:
            sorted_zcr = sorted(self._zcr_history)
            self._noise_floor_zcr = sorted_zcr[len(sorted_zcr) // 4]

        energy_above_threshold = energy > max(self.vad_energy_threshold, self._noise_floor_energy * 3.0)
        zcr_in_speech_range = (zcr > self.vad_zcr_threshold and zcr < 0.5)
        spectral_in_speech_range = (spectral_centroid > 0.15 and spectral_centroid < 0.7)

        speech_features = sum([energy_above_threshold, zcr_in_speech_range, spectral_in_speech_range])
        is_frame_speech = speech_features >= 2

        # ------------------------------------------------------------------
        # Voice analytics: pitch, tremor, pauses, speaking speed
        # ------------------------------------------------------------------
        self._total_frames += 1

        # Pitch estimation on normalized NumPy audio
        pitch_hz = self._estimate_pitch_hz(audio_np if audio_np is not None else audio_array)
        if pitch_hz > 0.0:
            self._pitch_history.append(pitch_hz)
            if self._last_pitch > 0.0:
                jitter = abs(pitch_hz - self._last_pitch) / max(self._last_pitch, 1e-3)
                self._pitch_jitter_history.append(jitter)
            self._last_pitch = pitch_hz

        # Track voiced / silence frames and speech segments
        if is_frame_speech:
            self._voiced_frames += 1
            if not self._prev_frame_was_speech:
                # New speech segment started
                self._speech_segment_count += 1
            self._prev_frame_was_speech = True
            self._speech_frames += 1
            self._silence_frames = 0
        else:
            self._prev_frame_was_speech = False
            self._silence_frames += 1
            self._speech_frames = 0

        # High-level features for confidence:
        # 1) Energy relative to threshold (already computed)
        denom = max(self.vad_energy_threshold * 3.0, 1e-4)
        energy_norm = float(np.clip(energy / denom, 0.0, 1.0))

        # 2) VAD feature agreement
        feature_agreement = speech_features / 3.0  # 0-1 based on how many features fired

        # 3) Pitch stability (lower variation => more stable/confident tone)
        pitch_stability = 0.5
        if len(self._pitch_history) >= 5:
            pitches = np.array(self._pitch_history, dtype=np.float32)
            mean_pitch = float(np.mean(pitches))
            std_pitch = float(np.std(pitches))
            if mean_pitch > 0.0:
                variation = std_pitch / mean_pitch  # relative variation
                # Map variation in [0, 0.4+] to stability [1, 0]
                pitch_stability = float(np.clip(1.0 - (variation / 0.4), 0.0, 1.0))

        # 4) Voice tremor (short-term jitter; more tremor => lower confidence)
        tremor_confidence = 1.0
        if len(self._pitch_jitter_history) >= 5:
            jitter_vals = np.array(self._pitch_jitter_history, dtype=np.float32)
            avg_jitter = float(np.mean(jitter_vals))
            # Map avg jitter in [0, 0.2+] to confidence [1, 0]
            tremor_confidence = float(np.clip(1.0 - (avg_jitter / 0.2), 0.0, 1.0))

        # 5) Pause ratio (more pauses => lower confidence)
        pause_ratio = 0.0
        if self._total_frames > 0:
            pause_ratio = float(1.0 - (self._voiced_frames / float(self._total_frames)))
        pause_confidence = float(np.clip(1.0 - (pause_ratio / 0.6), 0.0, 1.0))

        # 6) Speaking speed (speech segments per second; too low => less confident)
        frame_duration_sec = len(audio_array) / float(self.target_sample_rate or 16000)
        total_duration = self._total_frames * frame_duration_sec
        speaking_speed_conf = 0.5
        if total_duration > 0:
            segments_per_sec = self._speech_segment_count / total_duration
            # Map segments_per_sec in [0.5, 3.0] to [0, 1], clamp outside
            norm_rate = (segments_per_sec - 0.5) / (3.0 - 0.5)
            speaking_speed_conf = float(np.clip(norm_rate, 0.0, 1.0))

        # Combine all components into a single 0-1 confidence score
        confidence = float(np.clip(
            0.25 * energy_norm +
            0.20 * feature_agreement +
            0.20 * pitch_stability +
            0.20 * tremor_confidence +
            0.15 * pause_confidence +
            0.10 * speaking_speed_conf,
            0.0,
            1.0
        ))

        if self._speech_frames >= self._speech_threshold:
            self._is_speaking = True
        elif self._silence_frames >= self._silence_threshold:
            self._is_speaking = False

        # Convert to Int16 PCM
        if TF_AVAILABLE:
            pcm16 = (tensor.numpy() * 32767).astype(np.int16)
        else:
            pcm16 = (audio * 32767).astype(np.int16)

        return pcm16.tobytes(), self._is_speaking, confidence
    
    def convert_float32_to_pcm16(self, audio_data: bytes) -> bytes:
        """Convert Float32 audio to Int16 PCM."""
        audio_array = np.frombuffer(audio_data, dtype=np.float32)
        # Normalize if needed
        max_val = np.max(np.abs(audio_array))
        if max_val > 1.0:
            audio_array = audio_array / max_val
        pcm16 = (audio_array * 32767).astype(np.int16)
        return pcm16.tobytes()
    
    def convert_pcm16_to_float32(self, audio_data: bytes) -> bytes:
        """Convert Int16 PCM to Float32."""
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
        float32 = audio_array / 32768.0
        return float32.tobytes()
    
    def get_energy(self, audio_data: bytes, input_format: str = 'float32') -> float:
        """Get the energy level of audio data."""
        if input_format == 'float32':
            audio_array = np.frombuffer(audio_data, dtype=np.float32)
        else:
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        if len(audio_array) == 0:
            return 0.0

        if TF_AVAILABLE:
            tensor = tf.constant(audio_array, dtype=tf.float32)
            return float(tf.sqrt(tf.reduce_mean(tf.square(tensor))))
        return float(np.sqrt(np.mean(np.square(audio_array))))
    
    def is_speaking(self) -> bool:
        """Get current speaking state."""
        return self._is_speaking
    
    def reset(self):
        """Reset VAD and noise suppression state."""
        self._energy_history.clear()
        self._zcr_history.clear()
        self._noise_floor_energy = 0.01
        self._noise_floor_zcr = 0.1
        self._is_speaking = False
        self._speech_frames = 0
        self._silence_frames = 0
        self._noise_spectrum = None
        self._frame_count = 0


# Singleton instance for reuse
_processor_instance = None

def get_audio_processor() -> TFAudioProcessor:
    """Get or create the global audio processor instance."""
    global _processor_instance
    if _processor_instance is None:
        _processor_instance = TFAudioProcessor()
    return _processor_instance
