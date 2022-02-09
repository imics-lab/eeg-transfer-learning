### SPP-EEGNET: An Input-Agnostic Self-Supervised EEG Rrpresentation Model For Inter-Dataset Transfer Learning

---
The paper is published in The 18th International Conference on Computing and Information Technology (IC2IT)



---

**Abstract:** 
There is currently a scarcity of labeled Electroencephalography (EEG) recordings, and different datasets usually have
incompatible setups (e.g., various sampling rates, number of
channels, event lengths, etc.). These issues hinder machine
learning practitioners from training general-purpose EEG
models that can be reused on specific EEG classification
tasks through transfer learning. We present a deep convolutional neural network architecture with a spatial pyramid
pooling layer that is able to take in EEG signals of varying
dimensionality and extract their features to fixed-size vectors. The model is trained with a contrastive self-supervised
learning task on a large unlabelled dataset. We introduce a
set of EEG signal augmentation techniques to generate large
amounts of sample pairs to train the feature extractor. We
then transfer the trained feature extractor to new downstream
tasks. The experimental results show that the first few convolutional layers of the feature extractor learn the general
features of the EEG data, which can significantly improve the
classification performance on new datasets.




---

**Methodology:**



1.   SPP-EEG Feature extractor
2.   Contrastive Learning
3.   Transfer learning

---


**Code Structure:**

datasets/
> tuh_ssl_edf.py -- *tuh normal/abnormal dataset data loader used for self-supervised training*

> tuh_downstream_edf.py  -- *tuh normal/abnormal dataset data loader used for downstream transfer learning*

> EEGBCI_edf.py -- *EEGBCI motor imagery dataset data loader used for downstream transfer learning*


models/
> feature_extractor.py -- *SPP-EEG feature extractor model architecture*

> SSL_model.py -- *Contrastive learning model used for self-supervised learning*

> downstream_models.py -- *some downstream models used for transfer learning*


preprocessing/
> preprocesses.py -- *EEG signal propressing code for self-supervised training on the tuh normal/abnormal dataset*

> signalTransformation.py -- *EEG signal transformations used for generating augmenting EEG signals*

> tuh_downstream_preprocess.py -- *EEG signal proprocessing for downstram tuh normal/abnormal dataset*

examples_res/
> *.txt  -- *Recording names used for downstream transfer learning*

> *.pt -- *Pre-trained feature extractor*

> /tuh_tf_results *Raw transfer learning results for three extreme training sets*


images/
> EEGBCI.pdf -- *Transfer learning results on EEGBCI dataset*

> SPP-EEG.pdf -- *The feature extractor architecture*

> contrastive.pdf -- *The contrastive learning pipeline*

> TUHDownstream.pdf -- *Feature extractor transfer learning results on TUH dataset with 30 normal and 30 abnormal recordings*

> TUHDownstream_freeze0.pdf -- *Feature extractor transfer learning average results on TUH dataset with 5 normal and 5 abnormal recordings.*

> TUHDownstrean_freeze4.pdf -- *Feature extractor with the first layer parameters frozen  transfer learning average results on TUH dataset with 5 normal and 5 abnormal recordings.*

train_ssl.py -- *self-supervised training process on tuh normal/abnormal dataset*


tuh_downstream.py -- *tuh normal/abnormal dataset downstream transfer learning process* 


EEGBCI_downstream.py -- *EEGBCI motor imagery dataset downstream transfer learning process*


train_helpers.py -- *Some training helper functions*

tuh_tf_smallset.py -- *transfer learning process on extreme small tuh normal/abnormal dataset*

examples.ipynb -- *The example codes how to do conduct transfer learning on the pre-trained feature extractor*

---
About the datasets we used in this project:
1. TUH normal/abnormal dataset, you can it find from the link: https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml
Choose the option 'The TUH Abnormal EEG Corpus'. Note: You may need to sign up and send request to the organization to get the data. Due to the intellectual property reason, we can not directly share the data we used in the experiments.

2. EEGBCI Motor Imagery dataset, you can find it from the link: https://physionet.org/content/eegmmidb/1.0.0/

3. If reproduce the experiments, some paths settings in the code might need to be changed to correspond your local environment.

--- 
To train feature extractor on the self-supervised contrastive task:

```
python train_ssl.py -b 32 -e 100 --save-name='tuh_all_ckp'

```

To transfer trained feature extractor to tuh dataset

```
python tuh_downstream.py -e 50 --load-model='./example_res/tuh_all_ckp.pt' --layers=4
```

To transfer trained feature extractor to EEGBCI motor imagery dataset

```
python EEGBCI.py -e 50 --load-model='./example_res/tuh_all_ckp.pt' --task-number=1 --layers=4  
```

To train the feature extractor on EEGBCI motor imagery dataset from scratch
```
python EEGBCI.py -e 50 --task-number=1
```

To transfer trained feature extractor to the extreme small tuh dataset
```
python tuh_tf_smallset.py -e 50 --load-model='./example_res/tuh_all_ckp.pt' --layers=4 --freeze=0
```

To check example codes about how to load the pre-trained feature extractor and conduct transfer learning
```
examples.ipynb
```
More experiment visualization plots are shown in foder ./images

The raw loss and accuracy results of transfer learning on extreme small datasets are shown in folder ./examples_res/tuh_tf_results

Note: When we plot the transfer learning training and testing accuracy, we choose the epoch number where the model shows the smallest loss on the test set. 
