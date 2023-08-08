## Note
Implementation of other baselines can be found on [GIGN](https://github.com/guaguabujianle/GIGN).

## Dataset
All data used in this paper are publicly available at the following locations:
- **PDBbind v2016 and v2019:** [pdbbind](http://www.pdbbind.org.cn/download.php)
- **2013 and 2016 core sets:** [casf](http://www.pdbbind.org.cn/casf.php)

The preprocessed data can be downloaded from [Graphs](https://drive.google.com/file/d/1oGUP4z7htNXyxTqx95HNSDLsaoxa3fX7/view?usp=share_link).

## Requirements  
dgl==0.9.0  
networkx==2.5  
numpy==1.19.2  
pandas==1.1.5  
pymol==0.1.0  
rdkit==2022.3.5  
scikit_learn==1.1.2  
scipy==1.5.2  
torch==1.10.2  
tqdm==4.63.0  
openbabel==3.3.1 (conda install -c conda-forge openbabel)


## Descriptions of Folders and Files
- **`./data`:** Contains information about various datasets. Download and organize preprocessed datasets as described.
- **`./config`:** Parameters used in EHIGN.
- **`./log`:** Logger.
- **`./model`:** Contains model checkpoints and training records.
- **Scripts and Implementations:** Various Python files implementing models, preprocessing, training, and testing.

## Step-by-step Running

### 1. Model Training
- Download the preprocessed datasets and organize them in the `./data` folder.
- Run `python train.py`.

### 2. Model Testing
- Run `python test.py` (modify file paths in the source code if necessary).

### 3. Process Raw Data
- Run a demo using provided examples:
  - `python preprocess_complex.py`
  - `python graph_constructor.py`
  - `python train_example.py`

### 4. Test the Trained Model in Other External Test Sets
- Organize the data as described.  
  -data  
  &ensp;&ensp;-external_test  
  &ensp; &ensp;&ensp;&ensp; -pdb_id  
  &ensp; &ensp; &ensp;&ensp;&ensp;&ensp;-pdb_id_ligand.mol2  
  &ensp; &ensp; &ensp;&ensp;&ensp;&ensp;-pdb_id_protein.pdb  
  - Execute the following commands:
    - `python preprocess_complex.py`
    - `python graph_constructor.py`
    - `python test.py`
    - (Modify file paths in the source code if necessary)

### 5. Cold Start Settings
- Use datasets found in the `./cold_start_data` folder.
- Execute scripts `train_random.py`, `train_scaffold.py`, and `train_sequence.py` if the original training set has been processed.





## Descriptions of folders and files
+ **./data**: This folder contains information about train, valid, test2013, test2016, and test2019 data sets. You should first download the preprocessed datasets from https://drive.google.com/file/d/1oGUP4z7htNXyxTqx95HNSDLsaoxa3fX7/view?usp=share_link, and put them into this folder and organize them as './data/train', './data/valid', './data/test2013/', './data/test2016/', and  './data/test2019/'. We also provide a toy set with 50 examples to explain how to process raw data and train EHIGN model from scratch. 
+ **./config**: Parameters used in EHIGN.
+ **./log**: Logger.
+ **./model**: A folder contains model checkpoints and records about the training processing.
+ **CIGConv.py**: The implementation of CIGConv.
+ **NIGConv.py**: The implementation of NIGConv.
+ **EHIGN.py**: The implementation of EHIGN.
+ **HGC.py**: The implementation of the heterogeneous graph neural network, where most of the contents are copied from the source code of dgl, but we have made some modifications so that it can process edge features.
+ **preprocess_complex.py**: Prepare input complexes. The input ligand and protein should be .mol2 and .pdb formats, respectively. Proteins are first cropped around the ligand within 5 angstrom using pymol. Each protein and ligand is then processed into a protein-ligand complex using rdkit (i.e., a tuple that contains a ligand and a protein).
+ **graph_constructor.py**: Convert protein-ligand complexes into heterogeneous graphs.
+ **train.py**: Train EHIGN model.
+ **test.py**, Test a trained model on 2013 core set, 2016 core set, and 2019 holdout sets and print the results.
+ **performance.py**, Report the performance with three independent runs in mean (std) format. 
+ **utils.py** The File that includes useful tools for model training.

## Step-by-step running:  

### 1. Model training
Firstly, download the preprocessed datasets from https://drive.google.com/file/d/1oGUP4z7htNXyxTqx95HNSDLsaoxa3fX7/view?usp=share_link, and put them into this folder and organize them as './data/train', './data/valid', './data/test2013/', './data/test2016/', and  './data/test2019/'.  
Secondly, run train.py using `python train.py`.  

### 2. Model testing
Run test.py using `python test.py`.    
You may need to modify some file paths in the source code before running it.

### 3. Process raw data
We provide a demo to explain how to process the raw data. This demo use ./data/toy_examples.csv and ./data/toy_set/ as examples.  
Firstly, run preprocess_complex.py using `python preprocess_complex.py`.    
Secondly, run graph_constructor.py using `python graph_constructor.py`.  
Thirdly, run train.py using `python train_example.py`.    

### 4. Test the trained model in other external test sets
Firstly, please organize the data as a structure similar to './data/toy_set' folder.  
-data  
&ensp;&ensp;-external_test  
&ensp; &ensp;&ensp;&ensp; -pdb_id  
&ensp; &ensp; &ensp;&ensp;&ensp;&ensp;-pdb_id_ligand.mol2  
&ensp; &ensp; &ensp;&ensp;&ensp;&ensp;-pdb_id_protein.pdb  
Secondly, run preprocess_complex.py using `python preprocess_complex.py`.  
Thirdly, run graph_constructor.py using `python graph_constructor.py`.  
Fourth, run test.py using `python test.py`.  
You may need to modify some file paths in the source code before running it.  

### 5. Cold start settings
The datasets for the cold start settings can be found in the './cold_start_data' folder. These datasets are created from the original training set, taking into account structural differences. If you have already processed the original training set and placed it in the './data/train' folder, you can directly execute the 'train_random.py', 'train_scaffold.py', and 'train_sequence.py' scripts.


