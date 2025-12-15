import os
import shutil
from gradio_client import Client, handle_file


class RetrievalBasedVoiceConverter:
    def __init__(self, api_source="r3gm/RVC_ZERO"):
        """
        Initialize the RVC synthesizer.
        :param api_source: HuggingFace Space path
        """
        print(f"🔗 Initializing RVC Client: {api_source}...")
        self.client = Client(api_source)

    def run(
        self,
        audio_path,
        model_path,
        index_path,
        output_path,
        pitch_shift=0,
        index_rate=0.75,
    ):
        """
        Execute voice conversion.
        :param audio_path: Input audio file path (e.g., ./input.wav)
        :param model_path: .pth model file path
        :param index_path: .index index file path
        :param output_path: Output file save path
        :param pitch_shift: Pitch shift (male to female +12, female to male -12, same gender 0)
        :param index_rate: Index rate (affects voice timbre restoration)
        """

        # Check if files exist
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"❌ Input audio not found: {audio_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Model file not found: {model_path}")

        print(f"🎤 Starting conversion: {os.path.basename(audio_path)}")
        print("📤 Uploading to compute node...")

        try:
            # Call API
            result = self.client.predict(
                [handle_file(audio_path)],  # 1. audio_files
                handle_file(model_path),  # 2. file_m
                "rmvpe+",  # 3. pitch_alg
                pitch_shift,  # 4. pitch_lvl
                handle_file(index_path),  # 5. file_index
                index_rate,  # 6. index_inf
                3,  # 7. r_m_f
                0.25,  # 8. e_r
                0.5,  # 9. c_b_p
                False,  # 10. active_noise_reduce
                False,  # 11. audio_effects
                "wav",  # 12. type_output
                1,  # 13. steps
                api_name="/run",
            )

            # Handle return result (API may return list or single path string)
            source_file = result[0] if isinstance(result, list) else result

            print(f"✅ Conversion successful! Temp file: {source_file}")

            # Move result from temp to specified output path
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            shutil.copy(source_file, output_path)

            print(f"💾 File saved to: {output_path}")
            return output_path

        except Exception as e:
            print(f"❌ Conversion failed: {e}")
            raise e
