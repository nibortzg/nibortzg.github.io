---
title: (論文筆記) Designing Network Design Spaces -> Regnet
author:
  name: Nibor
  link: 
date: 2021-01-20 22:33:00 +0800
categories: [Computer Vision,Design Paragigm]
tags: [Computer Vision]
math: true
mermaid: true
image: 
  src: https://i.imgur.com/znfBRJe.png

  width: 800
  height: 500
---

## 前言: 
這篇論文試圖使用統計的方法來縮減design space，一步步的將AnyNet 縮減成RegNet, 並結總結出幾個有用的設計準則。雖然模型只在imagenet 上 train 10 epoch 讓公信力有點不足，不仍然值得嘗試。

## 方法:
![](https://i.imgur.com/ktzPdmv.png)
![](https://i.imgur.com/CxviXHo.png)

首先作者先定義了一種AnyNet, 而這個Anynet由以下幾種參數定義，每個stage的block 數量 $d_i$ ,每個stage寬度(channels) $w_i$ , bottleneck ratio $b_i$, group width $g_i$ (如果 $g_i == 1$ 則為depthwise convolution)


如何將AnyNet的搜索空間縮減過程就不深究，有興趣直接看一下論文，基本上都是作者對於EDF(Empirical distrubution function)的比較，然後選定優化方向。

在縮減過程中作者得到幾個結論

1. AnyNetX_B & AnyNetX_C 若所有的 $b_i = b$ 且 $g_i = g$, 所得到EDF的並不會變化,所以accuracy 並不會有所損失
2. AnyNetX_D & AnyNet_E,若 $w_{i+1} \geq w_i$ 且 $d_{i+1} \geq d_{i}$ 則可以讓整體精確度上升


![](https://i.imgur.com/rG6L2wP.png)
之後作者在AnyNet的Design space 中找出表現好得model，並且使用quentized linear parameterization (Linear fit) 試圖去總結出一些結論，然後他發現可以fit這條線越好的model有越好的performance.

並給出了RegNet fit 出來的結論
1. **最好的model 大約 20 stages (60 layers)**
2. **bottleneck ration = 1 是最好的, 移除 bottleneck !!** 
3. **Width mutltiplier 為 2.5 最佳, 跟常用的2 差不多**

複雜度分析:
activation 定義: the sizeof the output tensor of all conv layers
**Run time 與 activation 的關係比flops 更加緊密**
![](https://i.imgur.com/7TUgxWS.png)


![](https://i.imgur.com/f4kXhXv.png)
圖為最好的12個模型，可以看出在低parameters 時, 每個stage的 blocks 數目會逐漸上升，但是在大一些的模型，在stage 3 的blocks 數目會很多，但是stage 4 的數目會很少

Ablation study 結論:
1. inverted bottleneck 會使performance 下降
2. Swish  vs ReLU: swish 在低flops 時比較好，ReLU 在高flops 比較好, 且swish 在depthwise conv 時表現得比ReLU 好很多
3. SE 會增加精度
 