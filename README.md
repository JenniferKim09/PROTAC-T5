The code provided here is for the  **PROTAC-T5: Electron Cloud Representation Guided Proteolysis Targeting Chimera (PROTAC) Design with Multimodal Contrastive Learning and the Large Language Model** paper. First of all, prepare the environment:

- ```
git clone https://github.com/JenniferKim09/PROTAC-T5.git
cd PROTAC-T5
conda env create -f environment.yml -n protac-t5
conda activate protac-t5
  ```

## 1a. Direct download of data and ckpt

Datasets used for the training of the model (put under `./data`) and the trained checkpoint (put under `./ckpt`) can be downloaded [here](https://doi.org/10.5281/zenodo.20304396). 
  

## 1b. Manual preparation of the datasets
If you want to try the data preparation for your own data, use the code as:

- ```
python lig2ecloud.py 
  ```

The `.csv` file and `.sdf` of your data should be previously prepared and the path mentioned in `lig2ecloud.py` should be changed to your path. Users can refer to the examples `./data/PROTAC.csv` and `./data/PROTAC.sdf`. 


## 2. Train

* After the preparation of the datasets, the training can be performed by:

- ```
python train.py 
  ```


## 3. Generation

* Generate the linkers with the trained checkpoint:

- ```
python generate.py 
  ```
 

## 4. Combine the linkers with the fragments

* Refer to `combine_molecules.ipynb` to obtain the complete molecules. 


## 5. Evaluation

* The generated molecules can be evaluated under [this benchmark](https://github.com/JenniferKim09/PROTAC-benchmark) for all metrics mentioned in the paper. 











