## 資料準備

本計畫使用 Axolotl 框架進行大型語言模型的訓練，因此請遵守以下步驟：

1. `git clone https://github.com/axolotl-ai-cloud/axolotl`
2. 將 `training/configs` 資料夾複製到 `axolotl/` 中
3. 從雲端下載訓練資料 (data/) 下來後，將整個資料夾複製到 `axolotl/` 中

做完上述步驟，除了 Axolotl 本身的內容外，您應該還會有以下兩個資料夾在裡面：

- `configs/`: 存放訓練的設定檔
- `data/`: 存放訓練的資料

接著，您可以參考 `axolotl/README.md` 的說明進行環境安裝與訓練。

## Axolotl 環境安裝

建議使用 Conda 進行環境安裝，Axolotl 官方要求 Python 版本需 >= 3.10

1. `conda create -n axolotl python=3.10`
2. `conda activate axolotl`
3. Install pytorch stable https://pytorch.org/get-started/locally/
4. pip3 install packaging
5. pip3 install -e '.[flash-attn,deepspeed]'
6. 需要登入 huggingface，並請確定您的帳戶有在要使用的模型頁面上取得授權: `bash huggingface-cli login`

## 模型訓練

- 模型訓練：`accelerate launch -m axolotl.cli.train <模型訓練設定檔>`
  - 範例：`accelerate launch -m axolotl.cli.train configs/relation_extraction-chatml-qlora.yml`
- 合併 LoRA 權重: `python3 -m axolotl.cli.merge_lora <訓練 LoRA 模型時的 config 檔案> --lora_model_dir <LoRA 權重的存放位置>`
  - 範例：`python3 -m axolotl.cli.merge_lora configs/relation_extraction-chatml-qlora.yml --lora_model_dir exp/mistral/7b-instruct-v0.2/triplet_rationale_chatml_qlora/checkpoint-xxx/`
