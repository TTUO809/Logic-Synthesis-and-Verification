import sys
import torch
import platform

def run_test():
    print("="*50)
    print("🚀 DeepGate2 環境終極驗證程序")
    print("="*50)

    # 1. 基本系統資訊
    print(f"【系統資訊】")
    print(f"OS Platform: {platform.platform()}")
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"Current GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Runtime Version: {torch.version.cuda}")
    else:
        print("❌ 警告：未偵測到 CUDA ，請檢查驅動程式與 PyTorch 版本對位。")
    print("-" * 30)

    # 2. 核心 GNN 套件導入測試
    print(f"【GNN 套件導入測試】")
    try:
        import torch_scatter
        import torch_sparse
        import torch_cluster
        import torch_spline_conv
        import torch_geometric
        print("✅ torch-scatter, sparse, cluster, spline-conv: 導入成功")
    except ImportError as e:
        print(f"❌ 導入失敗：{e}")
        return

    # 3. 深度學習運算測試 (矩陣乘法)
    print("-" * 30)
    print(f"【GPU 運算測試】")
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.matmul(x, x)
        print(f"✅ GPU 矩陣乘法測試：成功 (結果維度: {y.shape})")
    except Exception as e:
        print(f"❌ GPU 運算失敗：{e}")

    # 4. GNN 特殊算子測試 (解決先前遇到的 undefined symbol 問題)
    try:
        # 測試 scatter_sum
        src = torch.randn(5, 4).cuda()                  # 5 個節點，每個節點 4 維特徵
        index = torch.tensor([0, 0, 1, 1, 2]).cuda()    # 節點分組索引
        out = torch_scatter.scatter_sum(src, index, dim=0)  # 按照 index 對 src 進行 sum 聚合
        print(f"✅ GNN CUDA 算子測試 (Scatter)：成功 (輸出維度: {out.shape})")
        
        # 測試 sparse 矩陣運算
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]).cuda()  # 邊的索引
        edge_attr = torch.randn(4).cuda()  # 邊的特徵
        adj = torch_sparse.SparseTensor(row=edge_index[0], col=edge_index[1], value=edge_attr)
        
        print(f"✅ GNN CUDA 算子測試 (Sparse)：成功 (邊數: {adj.nnz()})")
        print("\n🎉 恭喜！環境驗證完全通過，可以開始執行訓練任務。")
    except Exception as e:
        print(f"❌ GNN 算子測試失敗：{e}")
        print("\n💡 提示：若出現 'undefined symbol'，代表二進位檔版本仍不匹配。")

    print("="*50)

if __name__ == "__main__":
    run_test()