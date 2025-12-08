# GraphRec-WWW19

## GraphRec: Graph Neural Networks for Social Recommendation

## Abstract
In recent years, Graph Neural Networks (GNNs), which can naturally integrate node information and topological structure, have been demonstrated to be powerful in learning on graph data. These advantages of GNNs provide great potential to ad- vance social recommendation since data in social recommender systems can be represented as user-user social graph and user-item graph; and learning latent factors of users and items is the key. However, building social recommender systems based on GNNs faces challenges. For example, the user-item graph encodes both interactions and their associated opinions; social relations have heterogeneous strengths; users involve in two graphs (e.g., the user-user social graph and the user-item graph). To address the three aforementioned challenges simultaneously, in this paper, we present a novel graph neural network framework (GraphRec) for social recommendations. In particular, we provide a principled approach to jointly capture interactions and opinions in the user-item graph and propose the framework GraphRec, which coherently models two graphs and heterogeneous strengths. Extensive experiments on two real-world datasets demonstrate the effectiveness of the proposed framework GraphRec.

## Repository Structure & File Descriptions

### **1. Core Model Files**
* **`GraphRec.py`**: Main model architecture and training loop.
* **`Social_Encoders.py`** / **`Social_Aggregators.py`**: Handling user-user social graph aggregation.
* **`UV_Encoders.py`** / **`UV_Aggregators.py`**: Handling user-item interaction graph aggregation.

### **2. Dynamic Context-Aware (DCA) Extensions**
Files implementing our custom "Gated" attention mechanisms:
* **`UV_Aggregators_alpha.py`**: **DCA-α** (Item-Opinion Gate). Adapts fusion of item and opinion embeddings.
* **`Social_Aggregators_beta.py`**: **DCA-β** (Social Gate). Dynamically adjusts trust influence between users.
* **`UV_Aggregators_miu.py`**: **DCA-μ** (User-Item Trust Gate). Modulates how much an item "trusts" a user's opinion.

### **3. Data Processing**
* **`preprocess_movielens.py`**: Custom script to map **MovieLens 32M** users to the **Ciao** social graph.

## Data Processing
This project supports **Ciao**, **Epinions**, **Amazon**, and **MovieLens-Trust** dataset.

### **MovieLens Setup**
We construct a hybrid dataset by mapping users from **MovieLens 32M** to the social graph structure of **Ciao**.

1.  **Download Data:**
    * Get `ratings.csv` from the [MovieLens 32M Dataset](https://grouplens.org/datasets/movielens/).
    * Get `trustnetwork.txt` from the [Ciao Dataset](https://www.cse.msu.edu/~tangjili/trust.html).
2.  **Run Preprocessing:**
    ```bash
    python preprocess_movielens.py
    ```
    * *Input:* `ratings.csv`, `trustnetwork.txt`
    * *Output:* `movielens.pickle` (contains train/val/test splits, history lists, and adjacency maps).

### **Amazon Setup**
We construct a hybrid dataset by mapping users from **Amazon** to the social graph structure of **Ciao**.
Data: [Amazon Dataset](https://snap.stanford.edu/data/amazon-meta.html)

## Ablation Study
This module evaluates the importance of each GraphRec component.
Ablation variants implemented:
| Variant           | Description                                                 |
| ----------------- | ----------------------------------------------------------- |
| **GraphRec-SN**   | Removes the social graph → uses only item-space aggregation |
| **GraphRec+AC** | **Attention Context**: The baseline GraphRec model using standard attention mechanisms to aggregate context from neighbors. |
| **GraphRec+AG** | **Attention Gating**: Our proposed **DCA** model that replaces standard attention with learnable dynamic gates ($\alpha, \beta, \mu$) to filter noisy signals. |
| **No-α**          | Removes item aggregation attention                          |
| **No-β**          | Removes social aggregation attention                        |
| **No-μ**          | Removes user aggregation attention                          |
| **No-α&-β**  | Removes item aggregation attention &   Removes social aggregation attention                      |

Run baseline:
Unzip graphrec-ciao-epinions.zip
```
python run_GraphRec_complete.py
```
Run GraphRec-SN Ablation Study:
```
python GraphRec-SN.py
```

## Dynamic Context-Aware Attention (DCA)
We introduce learnable gates that adaptively modulate α, β, μ based on context, improving flexibility and handling noisy signals.

DCA Modules:
| Gate      | Purpose                                                     |
| --------- | ----------------------------------------------------------- |
| **DCA-α** | Personalize fusion of item embedding & opinion embedding    |
| **DCA-β** | Adjust social neighbor influence based on relation strength |
| **DCA-μ** | Let each item decide how much to trust each user            |

Run Dynamic Context-Aware α gate: Replace the original ```UV_Aggregators.py``` to ```UV_Aggregators_alpha.py```, and keep all of the other files unchanged.

Run Dynamic Context-Aware β gate: Replace the original ```Social_Aggregators.py``` to ```Social_Aggregators_beta.py```, and keep all of the other files unchanged.

Run Dynamic Context-Aware μ gate: Replace the original ```UV_Aggregators.py``` to ```UV_Aggregators_miu.py```, and keep all of the other files unchanged.

Run Dynamic Context-Aware α,β,μ gate combined: Replace the original ```Social_Aggregators.py``` to ```Social_Aggregators_beta.py```; Replace the original ```UV_Aggregators.py``` to ```UV_Aggregators_miu.py```and and uncomment the lines marked with ```#update```, and keep all of the other files unchanged.
## Introduction
 Graph Data in Social Recommendation. It contains two graphs including the user-item graph (left part) and the user-user social graph (right part). Note that the number on the edges of the user-item graph denotes the opinions (or rating score) of users on the items via the interactions.
![ 123](intro.png "Social Recommendations")


## Our Model GraphRec
The overall architecture of the proposed model. It contains three major components: user modeling, item modeling, and rating prediction.The first component is user modeling, which is to learn latent factors of users. As data in social recommender systems includes two different graphs, i.e., a social graph and a user-item graph, we are provided with a great opportunity to learn user representations from different perspectives. Therefore, two aggregations are introduced to respectively process these two different graphs. One is item aggregation, which can be utilized to understand users via interactions between users and items in the user-item graph (or item-space). The other is social aggregation, the relationship between users in the social graph, which can help model users from the social perspective (or social-space). Then, it is intuitive to obtain user latent factors by combining information from both item space and social space. The second component is item modeling, which is to learn latent factors of items. In order to consider both interactions and opinions in the user-item graph, we introduce user aggregation, which is to aggregate users’ opinions in item modeling. The third component is to learn model parameters via prediction by integrating user and item modeling components.

![ 123](GraphRec.png "GraphRec")


## Original Code

Author: Wenqi Fan (https://wenqifan03.github.io, email: wenqifan03@gmail.com) 

Also, I would be more than happy to provide a detailed answer for any questions you may have regarding GraphRec.

If you use this code, please cite our paper:
```
@inproceedings{fan2019graph,
  title={Graph Neural Networks for Social Recommendation},
  author={Fan, Wenqi and Ma, Yao and Li, Qing and He, Yuan and Zhao, Eric and Tang, Jiliang and Yin, Dawei},
  booktitle={The World Wide Web Conference},
  pages={417--426},
  year={2019},
  organization={ACM}
}
```

## Environment Settings
##### python: 3.6
##### pytorch: 0.2+

## Example to run the codes

Run GraphRec:
```
python run_GraphRec_example.py
```

Raw Datasets (Ciao and Epinions)  can be downloaded at [http://www.cse.msu.edu/~tangjili/trust.html](http://www.cse.msu.edu/~tangjili/trust.html)

## Deep Neural Networks for Social Recommendations

*  **<u>Wenqi Fan</u>**, Yao Ma , Qing Li, Jianping Wang, Guoyong Cai, Jiliang Tang, and Dawei Yin. **A Graph Neural Network Framework for Social Recommendations.** To appear in IEEE TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING (IEEE TKDE), 2020.

* **<u>Wenqi Fan</u>**, Yao Ma, Dawei Yin, Jianping Wang, Jiliang Tang, Qing Li.
  **Deep Social Collaborative Filtering.** In Proceedings of the 13th ACM Conference on Recommender Systems (RecSys 2019), 2019. (Long Paper,  Acceptance rate: 19%.) [[Arxiv](https://arxiv.org/abs/1907.06853)]    

* **<u>Wenqi Fan</u>**, Tyler Derr, Yao Ma, Jianping Wang, Jiliang Tang, Qing Li.
  **Deep Adversarial Social Recommendation.**  In Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI), 2019. [[Arxiv](https://arxiv.org/abs/1905.13160)]   [[Slides](https://drive.google.com/file/d/1lCvxGlkBm6ux3KderXlE0YE9ELSHlfbh/view?usp=sharing)]

* **<u>Wenqi Fan</u>**, Qing Li, Min Cheng. **Deep Modeling of Social Relations for Recommendation.**  In Proceedings of the Thirty-Second AAAI Conference on Artificial Intelligence. 2018. (Student Poster.)  [[PDF](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewPaper/16075)]



# Acknowledgements
The original version of this code base was from GraphSage. We owe many thanks to William L. Hamilton for making his code available. 
Please see the paper for funding details and additional (non-code related) acknowledgements.

Last Update Date: Oct, 2019
