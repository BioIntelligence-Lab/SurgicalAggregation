**NOTE:** This repository is currently work in progress.

# Surgical Aggregation: Federated Class-Heterogeneous Learning
### Pranav Kulkarni, Adway Kanhere, Paul H. Yi, Vishwa S. Parekh

The release of numerous chest x-ray datasets has spearheaded the development of deep learning models with expert-level performance. However, they have limited interoperability due to class-heterogeneity -- a result of inconsistent labeling schemes and partial annotations. Therefore, it is challenging to leverage these datasets in aggregate to train models with a complete representation of abnormalities that may occur within the thorax. In this work, we propose surgical aggregation, a federated learning framework for aggregating knowledge from class-heterogeneous datasets and learn a model that can simultaneously predict the presence of all disease labels present across the datasets. We evaluate our method using simulated and real-world class-heterogeneous datasets across both independent and identically distributed (iid) and non-iid settings. Our results show that surgical aggregation outperforms current methods, has better generalizability, and is a crucial first step towards tackling class-heterogeneity in federated learning to facilitate the development of clinically-useful models using previously non-interoperable chest x-ray datasets.

You can read our preprint [here](https://arxiv.org/abs/2301.06683)!
