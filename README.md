# SemanticBI
The source code of paper 'SemanticBI: quantifying intensities of transcription factor-DNA binding by learning from an ensemble of experiments'

## Dependency 
* Python 3.5.6
* Keras 2.2.2
* Tensorflow 1.9.0
* Sklearn 0.20.0

## PBM_66 experiment
1. Download data from [here](http://tools.genes.toronto.edu/deepbind/nbtcode/).
2. Copy files in *nbt3300-code/data/dream5/pbm/* to *SemanticBI/datasets/PBM_66/*. Files' paths are like *SemanticBI/datasets/PBM_66/sequences.tsv.gz* and *SemanticBI/datasets/PBM_66/targets.tsv.gz*.
3. Run PBM_66 experiment:
    ```
        python PBM66_train.py
    ```

## ChIP-seq_83 experiment
1. Download data from [here](http://tools.genes.toronto.edu/deepbind/nbtcode/).
2. Copy files in *nbt3300-code/data/encode/* to *SemanticBI/datasets/ChIP-seq_83/*. Files' paths are like *SemanticBI/datasets/ChIP-seq_83/ARID3A_HepG2_ARID3A_(NB100-279)_Stanford_AC.seq.gz* and *SemanticBI/datasets/ChIP-seq_83/ARID3A_HepG2_ARID3A_(NB100-279)_Stanford_B.seq.gz*.
3. New floder *SemanticBI/datasets/ChIP-seq_83/preprocessed_train/*.
4. Run ChIP-seq_83 experiment:
    ```
        python ChIPseq83_train.py
    ```

## Sharpr-MPRA experiment
1. Download data from [here](http://mitra.stanford.edu/kundaje/projects/mpra/data/).
2. Copy three .hdf5 files to *SemanticBI/datasets/Sharpr-MPRA/*. Files' paths are *SemanticBI/datasets/Sharpr-MPRA/test.hdf5*, *SemanticBI/datasets/Sharpr-MPRA/train.hdf5* and *SemanticBI/datasets/Sharpr-MPRA/valid.hdf5*.
3. Run PBM_66 experiment:
    ```
        python SharprMPRA_train.py
    ```
