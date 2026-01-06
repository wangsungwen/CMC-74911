# 執行 GFLOPs 分析腳本

關鍵注意事項
1. GFLOPs 與輸入尺寸相關： 理論計算複雜度 GFLOPs 會隨著您設定的 INPUT_SIZE（輸入影像解析度）而改變。請務必使用您在論文中進行主要評測時所用的解析度（例如 640x640），以確保數據的一致性。
2. MACs vs. FLOPs： 這是最容易混淆的地方。thop 套件計算的是 MACs（乘積累加運算）。在深度學習領域，通常近似認為 1 個 MAC 包含 1 次乘法和 1 次加法，因此 $1 \text{ MAC} \approx 2 \text{ FLOPs}$。所以，您需要將 thop 得到的 MACs 數量乘以 2，才能得到您要填入表格的 GFLOPs。
3. model.model： Ultralytics 的 YOLO() 物件是一個便利的包裝器。thop 需要的是底層的、純粹的 PyTorch nn.Module。這通常儲存在 .model 屬性中。

執行上述腳本後，您將在終端機中獲得一個清晰的 GFLOPs 數值（例如 理論 GFLOPs: 8.90），您可以將該數值直接填入「表 4.1」中對應 [需計算並增補] 的欄位。