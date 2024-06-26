# Illicit-transaction-address-detect-PU-Learning

Implementation of the paper [[arxiv](https://arxiv.org/abs/2303.02462), [inproceedings](https://ieeexplore.ieee.org/document/10174907)]:  

J. Luo, F. Poursafaei and X. Liu, "Towards Improved Illicit Node Detection with Positive-Unlabelled Learning," 2023 IEEE International Conference on Blockchain and Cryptocurrency (ICBC), Dubai, United Arab Emirates, 2023, pp. 1-5, doi: 10.1109/ICBC56567.2023.10174907.

(Here node means graph node, is a transaction address in a transaction network)

# Env

Use the `pip install -r requirements.txt` command to install all of the Python modules and packages listed.

# Data
Two public datasets:

Ethereum Phishing Transaction Network: [link](https://www.kaggle.com/datasets/xblock/ethereum-phishing-transaction-network)

Blockchain Elliptic Data Set: [link](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)

# Run

Run `python data_processor.py` to generate the csv files needed for the experiments. 

Ethereum Phishing Transaction Network data MulDiGraph.pkl is the example for the input dataset in the code.

Run `python main.py` to get the experiments results of three models: LR, Elkanoto PU, Bagging Pu.

Can choose other graph node embedding learning model from [karateclub](https://karateclub.readthedocs.io/en/latest/modules/root.html) (default: Role2Vec)


# Citation

Will be happy if this work is useful for your research.

```
@INPROCEEDINGS{luo_towards_nd_pu_icbc_2023,
  author={Luo, Junliang and Poursafaei, Farimah and Liu, Xue},
  booktitle={2023 IEEE International Conference on Blockchain and Cryptocurrency (ICBC)}, 
  title={Towards Improved Illicit Node Detection with Positive-Unlabelled Learning}, 
  year={2023},
  volume={},
  number={},
  pages={1-5},
  doi={10.1109/ICBC56567.2023.10174907}
}
```
