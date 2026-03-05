### ResShift使用的 Attention  
swin_transformer.py  
使用的是标准双向注意力计算, QK^T 没有mask  
它在同一窗口内允许当前位置看“前后所有 token”  