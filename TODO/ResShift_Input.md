### ResShift 输入输出  


#### Train  
gt 256x256 -> encoder 得到 latent 64x64, 在unet迭代中 加入残差噪声, 与目标lq latent 计算 loss
lq 64x64 -> 上采样 + encoder 得到latent y (256x256) 作为目标

encoder: 先将lq输入进行上采样指定倍数再进行encoder得到 latent  

#### Inference  
* lq 64x64 -> 上采样 + encoder 得到latent 256x256
* latent 64x64, timesteps, lq(条件图) 64x64 -> UNet 得到 新的latent. 其中 lq会通过 feature_extractor再与输入latent进行拼接  
* 将最终的输出latent进行decoder得到SR结果


#### encoder/decoder
latent倍数为4, 256x256 输入得到 64x64