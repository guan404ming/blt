import os
import shutil
from gradio_client import Client, handle_file

class RvcConverter:
    def __init__(self, api_source="r3gm/RVC_ZERO"):
        """
        初始化 RVC 合成器
        :param api_source: HuggingFace Space 的路徑
        """
        print(f"🔗 初始化 RVC Client: {api_source}...")
        self.client = Client(api_source)

    def run(self, audio_path, model_path, index_path, output_path, 
            pitch_shift=0, index_rate=0.75):
        """
        執行語音轉換
        :param audio_path: 輸入音訊的本地路徑 (例如 ./input.wav)
        :param model_path: .pth 模型檔案路徑
        :param index_path: .index 索引檔案路徑
        :param output_path: 輸出檔案的儲存路徑
        :param pitch_shift: 變調 (男轉女建議+12, 女轉男-12, 同性 0)
        :param index_rate: 索引率 (影響音色還原度)
        """
        
        # 1. 檢查檔案是否存在 (本機開發的安全防呆)
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"❌ 找不到輸入音訊: {audio_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ 找不到模型檔案: {model_path}")

        print(f"🎤 開始轉換: {os.path.basename(audio_path)}")
        print("📤 正在上傳至運算節點...")

        try:
            # 呼叫 API
            result = self.client.predict(
                [handle_file(audio_path)], # 1. audio_files
                handle_file(model_path),   # 2. file_m
                "rmvpe+",                  # 3. pitch_alg
                pitch_shift,               # 4. pitch_lvl
                handle_file(index_path),   # 5. file_index
                index_rate,                # 6. index_inf
                3,                         # 7. r_m_f
                0.25,                      # 8. e_r
                0.5,                       # 9. c_b_p
                False,                     # 10. active_noise_reduce
                False,                     # 11. audio_effects
                "wav",                     # 12. type_output
                1,                         # 13. steps
                api_name="/run"
            )

            # 處理回傳結果
            # result 根據 API 可能回傳 list 或單一字串路徑
            source_file = result[0] if isinstance(result, list) else result
            
            print(f"✅ 轉換成功！雲端暫存檔: {source_file}")

            # 將結果從暫存區移動到指定輸出路徑
            # 確保輸出資料夾存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            shutil.copy(source_file, output_path)
            
            print(f"💾 檔案已儲存至: {output_path}")
            return output_path

        except Exception as e:
            print(f"❌ 轉換失敗: {e}")
            raise e

# ================= 測試區塊 =================
# 這段代碼只有當你直接執行此檔案時才會跑 (python voice_synthesizer.py)
# 被其他程式 import 時不會跑，這是 Python 的標準寫法
if __name__ == "__main__":
    # 設定測試用的假路徑 (請替換成你 Mac 上的真實路徑)
    # 建議把模型檔案放在專案裡的某個資料夾，例如 tests/fixtures/ 或是本機的下載區
    
    # 範例路徑 (請自行修改)
    TEST_AUDIO = "/Users/georgecheng/Desktop/碩士班/深度學習於音樂分析及生成/MIR_project/audio.wav" 
    TEST_MODEL = "/Users/georgecheng/Desktop/碩士班/深度學習於音樂分析及生成/MIR_project/歌手model/統神 - Weights Model/model.pth"
    TEST_INDEX = "/Users/georgecheng/Desktop/碩士班/深度學習於音樂分析及生成/MIR_project/歌手model/統神 - Weights Model/model.index"
    TEST_OUTPUT = "./Users/georgecheng/Desktop/output_test.wav"

    if os.path.exists(TEST_AUDIO) and os.path.exists(TEST_MODEL):
        synthesizer = RvcConverter()
        synthesizer.run(TEST_AUDIO, TEST_MODEL, TEST_INDEX, TEST_OUTPUT)
    else:
        print("⚠️ 測試模式跳過：請設定下方的 TEST_AUDIO 與 TEST_MODEL 路徑來進行測試")