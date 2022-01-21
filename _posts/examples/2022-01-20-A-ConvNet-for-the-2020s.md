---
title: (論文筆記) A ConvNet for the 2020s 
author:
  name: Nibor
  link: 
date: 2021-01-20 22:33:00 +0800
categories: [Computer Vision]
tags: [Computer vision]
math: true
mermaid: true
image:
  src: https://i.imgur.com/ddJGZsj.png

  width: 800
  height: 500
---

## 前言: 
近幾年vision transformer 在各種會議上大放異彩, 隨著演進, 我們也可以發現transformer 引進越來越多跟CNN 相關的prior, 像是swin transformer的local window 就很有ConvNet的味道。觀察中可以發現vision transformer 和 CNN 變得越來越像, 且在一些下游任務中, 這些內建於CNN之中的bias會使得任務更好完成, 例如translation equivariance 對於object detection任務而言就很重要。

第二， 對於現在vision transformer 成功的關鍵，是在於transformer 本身，還是training techniques與一些其餘組件的改動，造成整體performance 的提升也是值得思考得問題，例如專精於training techniques 提昇讓Resnet 復活的論文[`ResNet strikes back: An improved training procedure in timm`](https://arxiv.org/pdf/2110.00476.pdf) 或是 [`Patches are all you need`](https://openreview.net/pdf?id=TVHS5Y4dNvM) 討論patches是否才是成功關鍵，都是值得思考的問題。

作者在此篇論文像是抄作業一般將他認為swin transformer 勝過於目前Resnet的關鍵複製過去，一步一步的 "modernizing" Resnet.

## 方法:
### 1. 訓練技術改進(Training techniques)
在前面提到timm 那篇論文中有提到光靠訓練技術(data augumentation, optimizer, training receipts)的進步就可以將Resnet50 上調到80.4, 但作者在這邊為了方便與swin transformer 比較, 使用與他們相近的訓練技術。光是這樣performance 就從76.1% -> 78.8% 有了2.7% 的進步。


### 2. 改變stage 的compute ratio
將原本Resnet 每個stage 的block 排列(3,4,6,3) 改成跟 Swin-T 一樣比例的 (3,3,9,3)也提升了performance。 值得注意的是，block 數目也從16 -> 18。 感覺performance 提升在這裡是跟多運算比較有關係。不過運算的前後分配也是一個值得思考的點。作者也給了兩篇參考文獻: [On network design spaces for visual recognition](https://arxiv.org/pdf/1905.13214.pdf) 和 [Designing network design spaces](https://arxiv.org/pdf/2003.13678.pdf).


### 3. 將stem 改成patch 形式
Stem 就是圖片進到CNN的第一層結構, 由於一般圖片都有很多冗餘資訊，所以基本上會採用很激進的dwonsample 策略，將input image縮成適當比例的feature map 再做操作。原本的Resnet 是使用7x7 conv with stride 2與max pooling 連結達到4倍 downsampling 的結果, 這裡使用激進的patchify做法，也就是使用large kernel size with non-overlapping convolution. 在此作者使用 4x4 convolution with stride 4.

```python
 stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
```


### 4. Bottleneck 設計 -> ResNeXt-ify
ResNext 的設計宗旨就是用更多的group 來換取更多的width。 在此作者使用MobileNet 所使用的 depthwise convolution (group=channel 的group convolution). 可以注意的是depthwise convolution 很像self-attention中的加權和，他們都是只把spatial dimension 的資訊混合而沒有讓資訊在channel中流通。

### 5. Bottleneck 設計 -> inverted bottleneck
![](https://i.imgur.com/BfSDCFg.png)
跟原本Mobilenet V2 的inverted bottleneck (b)相比, 作者這邊先把 depthwise convolution 位置從中間調整到最上面，主要是為了減少計算量，再者他將kernel size 變大成7x7, 實驗結果表示在7x7 之後更大的kernel size 並沒有將performance 提升

### 6. ReLU -> GELU
![](https://i.imgur.com/mDEqJ1z.png)
![](https://i.imgur.com/PEbuicE.png)
GELU 是ReLU的平滑版本, 在很多transformer 中都有使用像是Bert, GPT-2
作者在此使用並沒有讓準確度提升, 在實際應用上我應該也會傾向使用ReLU畢竟速度快又有比較多的library 支援

### 7. 更少的activation function 與 normalization layer
![](https://i.imgur.com/2XWIHZd.png)
如圖所示, 跟一般操作不同的是, 不一定要在每個 convolution layer後面加上activation function and BN.

### 8. 將BN換成LN (Layer Normalization)
BN 在ConvNet中可以穩定training 也可以減少過擬合, 然而近期的研究指出BN也會讓performance 降低, 且LN 大量的被用在transformer中。所以作者決定將BN換成LN。 
(注:直接將Resnet50 中的BN 換成LN會使得performance 下降)
![](https://i.imgur.com/VgNz2a9.png)

### 9. Downsampling layer 改成patch 形式

```python
downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
```

若直接將downsampling layer 改成non-overlapping 的convolution, CNN會發散，所以要加入 normalization layers 來穩定training 


## 重要結論:
> 以前認為Vision transformer 有比較少 inductive bias(prior) 所以在有大量pretrain data的情況下可以學得比較好， 但是ConvNext 實驗證明， 好好設計的ConvNet 並不會輸。
