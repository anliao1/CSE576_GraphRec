# GraphRec

## Beyond GraphRec: Can Graph Neural Networks Step Further for Multifaceted Social Recommendation?

## Abstract
GraphRec models social recommendation through three attention pathways over item interactions (𝛼), social neighbors (𝛽), and user–opinion features (𝜇). However, the necessity and robustness of these components remain unclear, particularly under varying graph fidelity. We revisit GraphRec with extensive ablation studies and propose Dynamic Context-Aware Attention (DCA), a lightweight gating extension that adaptively modulates the three pathways based on local context. Our ablations reveal that each pathway con- tributes differently across datasets: social aggregation (𝛽) is most influential on Ciao, while opinion aggregation (𝜇) plays a larger role on Epinions; and static attention proves vulnerable to sparse or noisy relations. Our proposed DCA extension mitigates these issues by amplifying reliable signals and down-weighting inconsistent ones. Across four datasets, DCA is validated to improve accuracy and stability, with the combined DCA-ALL model achieving the strongest overall performance.

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

Run Dynamic Context-Aware α,β,μ gate combined: Replace the original ```Social_Aggregators.py``` to ```Social_Aggregators_beta.py```; Replace the original ```UV_Aggregators.py``` to ```UV_Aggregators_miu.py```and  uncomment the lines marked with ```#update```, and keep all of the other files unchanged.
## Introduction
 Graph Data in Social Recommendation. It contains two graphs including the user-item graph (left part) and the user-user social graph (right part). Note that the number on the edges of the user-item graph denotes the opinions (or rating score) of users on the items via the interactions.
![ 123](intro.png "Social Recommendations")


## Our Model GraphRec
The overall architecture of the proposed model. It contains three major components: user modeling, item modeling, and rating prediction.The first component is user modeling, which is to learn latent factors of users. As data in social recommender systems includes two different graphs, i.e., a social graph and a user-item graph, we are provided with a great opportunity to learn user representations from different perspectives. Therefore, two aggregations are introduced to respectively process these two different graphs. One is item aggregation, which can be utilized to understand users via interactions between users and items in the user-item graph (or item-space). The other is social aggregation, the relationship between users in the social graph, which can help model users from the social perspective (or social-space). Then, it is intuitive to obtain user latent factors by combining information from both item space and social space. The second component is item modeling, which is to learn latent factors of items. In order to consider both interactions and opinions in the user-item graph, we introduce user aggregation, which is to aggregate users’ opinions in item modeling. The third component is to learn model parameters via prediction by integrating user and item modeling components.

![ 123](GraphRec.png "GraphRec")


## Original Code and Most Related Work
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
##### pytorch: 0.2
