

[TOC]

# Introduction

Graph neural network, as a powerful graph representation learning method, has been widely used in diverse scenarios, such as NLP, CV, and recommender systems.

As far as I can see, graph mining is highly related to recommender systems. Recommend one item to one user actually is the link prediction on the user-item graph.

This repository mainly consists of three parts:

- Graph Neural Network
- GNN based Recommendation
- GNN related Resources

We also have an Wechat Official Account, providing some materials about GNN and Recommendation.

![图与推荐](README.assets/图与推荐.png)

You're most welcome to join us with any contributions for GNN and Recommendation!
Here is the template for contributors:
```
[ID] Authors. **Paper_Name**. Conference&Year. [Paper](Paper_Link)
```
A simple example for template:
```
1. Long, Qingqing and Jin, Yilun and Song, Guojie and Li, Yi and Lin, Wei. **Graph Structural-topic Neural Network**. KDD 2020. [paper](https://arxiv.org/abs/2006.14278)
```
# Graph Neural Network

1. Giannis Nikolentzos and Michalis Vazirgiannis.**Random Walk Graph Neural Networks**. NeurIPS 2020.[paper](https://www.lix.polytechnique.fr/~nikolentzos/files/rw_gnns_neurips20)
2. Nicolas Keriven and Alberto Bietti and Samuel Vaiter. **Convergence and Stability of Graph Convolutional Networks on Large Random Graphs**. NeurIPS 2020. [paper](https://arxiv.org/abs/2006.01868) 
3. Nikolaos Karalias and Andreas Loukas. **Erdos Goes Neural: an Unsupervised Learning Framework for Combinatorial Optimization on Graphs**. NeurIPS 2020. [paper](https://arxiv.org/abs/2006.10643)
4. Xiang Zhang and Marinka Zitnik. **GNNGuard: Defending Graph Neural Networks against Adversarial Attacks**. NeurIPS 2020. [paper](https://arxiv.org/abs/2006.08149)
5. Zheng Ma and Junyu Xuan and Yu Guang Wang and Ming Li and Pietro Lio. **Path Integral Based Convolution and Pooling for Graph Neural Networks** NeurIPS 2020. [paper](https://arxiv.org/abs/2006.16811)

<details> 
<summary> more </summary> 
    
6. Daniel D. Johnson and Hugo Larochelle and Daniel Tarlow. **Learning Graph Structure With A Finite-State Automaton Layer**. NeurIPS 2020. [paper](https://arxiv.org/abs/2007.04929)
7. Vitaly Kurin and Saad Godil and Shimon Whiteson and Bryan Catanzaro. **Improving SAT Solver Heuristics with Graph Networks and Reinforcement Learning**. NeurIPS 2020. [paper](https://arxiv.org/abs/1909.11830)
8. Zhiwei Deng and Karthik Narasimhan and Olga Russakovsky. **Evolving Graphical Planner: Contextual Global Planning for Vision-and-Language Navigation** NeurIPS 2020. [paper](https://arxiv.org/abs/2007.05655)
9. Long, Qingqing and Jin, Yilun and Song, Guojie and Li, Yi and Lin, Wei. **Graph Structural-topic Neural Network**. KDD 2020. [paper](https://arxiv.org/abs/2006.14278)
10. Zang, Chengxi and Wang, Fei. **Neural Dynamics on Complex Networks** KDD2020. [paper](https://arxiv.org/abs/1908.06491)
11. Ganqu Cui, Jie Zhou, Cheng Yang, Zhiyuan Liu. Adaptive Graph Encoder for Attributed Graph Embedding KDD 2020. [paper](https://arxiv.org/pdf/2007.01594.pdf)
12. Dynamic Deep Neural Networks: Optimizing Accuracy-Efficiency Trade-offs by Selective Execution. AAAI 2018
13. Dynamic Network Embedding by Modeling Triadic Closure Process. AAAI 2018
14. DepthLGP: Learning Embeddings of Out-of-Sample Nodes in Dynamic Networks. AAAI 2018
15. A Generative Model for Dynamic Networks with Applications. AAAI 2019
16. Communication-optimal distributed dynamic graph clustering. AAAI 2019
17. EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs. AAAI 2020
18. Dynamic Network Pruning with Interpretable Layerwise Channel Selection. AAAI 2020
19. DyRep: Learning Representations over Dynamic Graphs. ICLR 2019
20. Dynamic Graph Representation Learning via Self-Attention Networks. ICLR 2019
21. The Logical Expressiveness of Graph Neural Networks. ICLR 2020 
22. Fast and Accurate Random Walk with Restart on Dynamic Graphs with Guarantees. WWW 2018
23. Dynamic Network Embedding : An Extended Approach for Skip-gram based Network Embedding. IJCAI 2018
24. Deep into Hypersphere: Robust and Unsupervised Anomaly Discovery in Dynamic Networks. IJCAI 2018
25. AddGraph: Anomaly Detection in Dynamic Graph using Attention-based Temporal GCN. IJCAI 2019
26. Network Embedding and Change Modeling in Dynamic Heterogeneous Networks. SIGIR 2019
27. Learning Dynamic Node Representations with Graph Neural Networks. SIGIR 2020
28. Dynamic Link Prediction by Integrating Node Vector Evolution and Local Neighborhood Representation. SIGIR 2020
29. NetWalk: A Flexible Deep Embedding Approach for Anomaly Detection in Dynamic Networks. KDD 2018
30. Fast and Accurate Anomaly Detection in Dynamic Graphs with a Two-Pronged Approach. KDD 2019
31. Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks. KDD 2019
32. Laplacian Change Point Detection for Dynamic Graphs. KDD 2020
33. Dynamic Heterogeneous Graph Neural Network for Real-time Event Prediction
    Neural Dynamics on Complex Networks KDD 2020
34. Fast Approximate Spectral Clustering for Dynamic Networks. ICML 2018
35. Improved Dynamic Graph Learning through Fault-Tolerant Sparsification. ICML 2019
36. Efficient SimRank Tracking in Dynamic Graphs. ICDE 2018
37. On Efficiently Detecting Overlapping Communities over Distributed Dynamic Graphs. ICDE 2018
38. Computing a Near-Maximum Independent Set in Dynamic Graphs. ICDE 2019
39. Finding Densest Lasting Subgraphs in Dynamic Graphs: A Stochastic Approach. ICDE 2019
40. Tracking Influential Nodes in Time-Decaying Dynamic Interaction Networks. ICDE 2019
41. Adaptive Dynamic Bipartite Graph Matching: A Reinforcement Learning Approach. ICDE 2019
42. A Fast Sketch Method for Mining User Similarities Over Fully Dynamic Graph Streams. 
43. Ziniu Hu, Yuxiao Dong, Kuansan Wang, Yizhou Sun. **Heterogeneous Graph Transformer.** WWW 2020
44. Yuxiang Ren and Bo Liu and Chao Huang and Peng Dai and Liefeng Bo and Jiawei Zhang. **Heterogeneous Deep Graph Infomax.** AAAI 2020
45. Xingyu Fu, Jiani Zhang, Ziqiao Meng, Irwin King. **Metapath Aggregated Graph Neural Network for Heterogeneous Graph Embedding.** WWW2020
46. Seongjun Yun, Minbyul Jeong, Raehyun Kim, Jaewoo Kang, Hyunwoo J. Kim. **Graph Transformer Networks.** NIPS 2019
47. Yuxin Xiao, Zecheng Zhang, Carl Yang, and Chengxiang Zhai. **Non-local Attention Learning on Large Heterogeneous Information Networks** IEEE Big Data 2019.
48. Shaohua Fan, Junxiong Zhu, Xiaotian Han, Chuan Shi, Linmei Hu, Biyu Ma, Yongliang Li.  KDD 2019. [paper](https://dl.acm.org/citation.cfm?id=3330673)
49. Chuxu Zhang, Dongjin Song, Chao Huang, Ananthram Swami, Nitesh V. Chawla. **Heterogeneous Graph Neural Network.** KDD 2019
50. Hao Peng, Jianxin Li, Qiran Gong, Yangqiu Song, Yuanxing Ning, Kunfeng Lai  and Philip S. Yu  IJCAI 2019. [paper](https://arxiv.org/abs/1906.04580)
51. Xiao Wang, Houye Ji, Chuan Shi, Bai Wang, Peng Cui, Philip S. Yu, Yanfang Ye. WWW 2019. [paper](https://github.com/Jhy1993/HAN)
52. Yizhou Zhang, Yun Xiong, Xiangnan Kong, Shanshan Li, Jinhong Mi, Yangyong Zhu.  WWW 2018. [paper](https://dl.acm.org/citation.cfm?id=3186106)
53. Ziqi Liu, Chaochao Chen, Xinxing Yang, Jun Zhou, Xiaolong Li, Le Song.  CIKM 2018.   [paper](https://dl.acm.org/citation.cfm?id=3272010)
54. Marinka Zitnik, Monica Agrawal, Jure Leskovec.  ISMB 2018 
55. Hao Yuan, Jiliang Tang, Xia Hu, Shuiwang Ji. **XGNN: Towards Model-Level Explanations of Graph Neural Networks** KDD2020. [paper](https://arxiv.org/pdf/2006.02587.pdf)
56. Lei Yang, Qingqiu Huang, Huaiyi Huang, Linning Xu, and Dahua Lin**Learn to Propagate Reliably on Noisy Affinity Graphs** ECCV2020. [paper](https://arxiv.org/pdf/2007.08802.pdf)
57. Yao Ma, Ziyi Guo, Zhaochun Ren, Eric Zhao, Jiliang Tang, Dawei Yin. **Streaming Graph Neural Networks**  SIGIR2020. [paper](https://arxiv.org/abs/1810.10627)

</details>


# GNN based Recommendation

1. Rex Ying, Ruining He, Kaifeng Chen, Pong Eksombatchai, William L. Hamilton, Jure Leskovec.  **Graph Convolutional Neural Networks for Web-Scale Recommender Systems.** KDD 2018. [paper](https://arxiv.org/abs/1806.01973)
2. Federico Monti, Michael M. Bronstein, Xavier Bresson. **Geometric Matrix Completion with Recurrent Multi-Graph Neural Networks.** NIPS 2017. [paper](https://arxiv.org/abs/1704.06803)
3. Rianne van den Berg, Thomas N. Kipf, Max Welling. **Graph Convolutional Matrix Completion.** 2017. [paper](https://arxiv.org/abs/1706.02263)
4. Jiani Zhang, Xingjian Shi, Shenglin Zhao, Irwin King. **STAR-GCN: Stacked and Reconstructed Graph Convolutional Networks for Recommender Systems.** IJCAI 2019. [paper](https://arxiv.org/pdf/1905.13129.pdf)
5. Haoyu Wang, Defu Lian, Yong Ge. **Binarized Collaborative Filtering with Distilling Graph Convolutional Networks.** IJCAI 2019. [paper](https://arxiv.org/pdf/1906.01829.pdf)

<details> 
<summary> more </summary> 

6. Chengfeng Xu, Pengpeng Zhao, Yanchi Liu, Victor S. Sheng, Jiajie Xu, Fuzhen Zhuang, Junhua Fang, Xiaofang Zhou. **Graph Contextualized Self-Attention Network for Session-based Recommendation.** IJCAI 2019. [paper](https://www.ijcai.org/proceedings/2019/0547.pdf)
7. Shu Wu, Yuyuan Tang, Yanqiao Zhu, Liang Wang, Xing Xie, Tieniu Tan. **Session-based Recommendation with Graph Neural Networks.** AAAI 2019. [paper](https://arxiv.org/pdf/1811.00855.pdf)
8. Jin Shang, Mingxuan Sun. **Geometric Hawkes Processes with Graph Convolutional Recurrent Neural Networks.** AAAI 2019. [paper](https://jshang2.github.io/pubs/geo.pdf)
9. Hongwei Wang, Fuzheng Zhang, Mengdi Zhang, Jure Leskovec, Miao Zhao, Wenjie Li, Zhongyuan Wang. **Knowledge-aware Graph Neural Networks with Label Smoothness Regularization for Recommender Systems.** KDD 2019. [paper](https://arxiv.org/pdf/1905.04413)
10. Yu Gong, Yu Zhu, Lu Duan, Qingwen Liu, Ziyu Guan, Fei Sun, Wenwu Ou, Kenny Q. Zhu. **Exact-K Recommendation via Maximal Clique Optimization.** KDD 2019. [paper](https://arxiv.org/pdf/1905.07089)
11. Xiang Wang, Xiangnan He, Yixin Cao, Meng Liu, Tat-Seng Chua. **KGAT: Knowledge Graph Attention Network for Recommendation.** KDD 2019. [paper](https://arxiv.org/pdf/1905.07854)
12. Hongwei Wang, Miao Zhao, Xing Xie, Wenjie Li, Minyi Guo. **Knowledge Graph Convolutional Networks for Recommender Systems.** WWW 2019. [paper](https://arxiv.org/pdf/1904.12575.pdf)
13. Qitian Wu, Hengrui Zhang, Xiaofeng Gao, Peng He, Paul Weng, Han Gao, Guihai Chen. **Dual Graph Attention Networks for Deep Latent Representation of Multifaceted Social Effects in Recommender Systems.** WWW 2019. [paper](https://arxiv.org/pdf/1903.10433.pdf)
14. Wenqi Fan, Yao Ma, Qing Li, Yuan He, Eric Zhao, Jiliang Tang, Dawei Yin. **Graph Neural Networks for Social Recommendation.** WWW 2019. [paper](https://arxiv.org/pdf/1902.07243.pdf)
15. Chen Ma, Liheng Ma, Yingxue Zhang, Jianing Sun, Xue Liu, Mark Coates. **Memory Augmented Graph Neural Networks for Sequential Recommendation.** AAAI 2020. [paper](https://arxiv.org/abs/1912.11730)
16. Lei Chen, Le Wu, Richang Hong, Kun Zhang, Meng Wang. **Revisiting Graph based Collaborative Filtering: A Linear Residual Graph Convolutional Network Approach.** AAAI 2020. [paper](https://arxiv.org/abs/2001.10167)
17. Muhan Zhang, Yixin Chen. **Inductive Matrix Completion Based on Graph Neural Networks.** ICLR 2020. [paper](https://openreview.net/pdf?id=ByxxgCEYDS)
18. Xueya Zhang, Tong Zhang, Xiaobin Hong, Zhen Cui, and Jian Yang. **Graph Wasserstein Correlation Analysis for Movie Retrieval** ECCV 2020. [paper](https://arxiv.org/pdf/2008.02648.pdf)
19. Xiaowei Jia , Handong Zhao , Zhe Lin , Ajinkya Kale , Vipin Kumar. **Personalized Image Retrieval with Sparse Graph Representation Learning** KDD2020. [paper](https://dl.acm.org/doi/pdf/10.1145/3394486.3403324)
20. Tianwen Chen, Raymond Chi-Wing Wong. **Handling Information Loss of Graph Neural Networks for Session-based Recommendation** KDD 2020 [paper](https://dl.acm.org/doi/pdf/10.1145/3394486.3403170)
21. Jianxin Chang, Chen Gao, Xiangnan He, Yong Li, Depeng Ji. **Bundle Recommendation with Graph Convolutional Networks** SIGIR2020. [paper](https://arxiv.org/pdf/2005.03475.pdf)
22. Chang-You Tai, Meng-Ru Wu, Yun-Wei Chu, Shao-Yu Chu, Lun-Wei Ku. **MVIN: Learning Multiview Items for Recommendation** SIGIR2020. [paper](https://arxiv.org/pdf/2005.12516.pdf)
23. Xingchen Li, Xiang Wang, Xiangnan He, Long Chen, Jun Xiao, Tat-Seng Chua. **Hierarchical Fashion Graph Network for Personalized Outfit Recommendation** SIGIR2020 [paper](https://arxiv.org/pdf/2005.12566.pdf)
24. Kelong Mao, Xi Xiao, Jieming Zhu, Biao Lu, Ruiming Tang, Xiuqiang He. **Item Tagging for Information Retrieval: A Tripartite Graph Neural Network based Approach** SIGIR2020. [paper](https://arxiv.org/pdf/2008.11567.pdf)
25. Le Wu, Yonghui Yang, Lei Chen, Defu Lian, Richang Hong, Meng Wang. **Learning to Transfer Graph Embeddings for Inductive Graph based Recommendation** SIGIR2020 [paper](https://arxiv.org/pdf/2005.11724.pdf)
26. Shijie Zhang, Hongzhi Yin, Tong Chen, Quoc Viet Nguyen Hung, Zi Huang, Lizhen Cui. **GCN-Based User Representation Learning for Unifying Robust Recommendation and Fraudster Detection** SIGIR2020 [paper](https://arxiv.org/pdf/2005.10150.pdf)
27. Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yongdong Zhang, Meng Wang.  **LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation** SIGIR2020 [paper](http://staff.ustc.edu.cn/~hexn/papers/sigir20-LightGCN.pdf)
28. Shen Wang, Jibing Gong, Jinlong Wang, Wenzheng Feng, Hao Peng, Jie Tang, Philip S. Yu.  **Attentional Graph Convolutional Networks for Knowledge Concept Recommendation in MOOCs in a Heterogeneous View** SIGIR2020 [paper](https://keg.cs.tsinghua.edu.cn/jietang/publications/Sigir20-Gong-et-al-MOOC-concept-recommendation.pdf)

</details>

# GNN related Resouces

## Class

Class for new people who are interested in GNN and Recommendation.

Link:  https://www.epubit.com/courseDetails?id=PCC72369cd0eb9e7



![图机器学习（时下最炙手可热新技术/8章3大模型应用）](README.assets/resize,m_fill,h_300,w_400,limit_0.png)

## Meterials

https://zhuanlan.zhihu.com/c_1158788280744173568

## QQ & Wechat

![image-20200925152306468](README.assets/image-20200925152306468.png)

