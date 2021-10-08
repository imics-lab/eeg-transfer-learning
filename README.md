### SPP-EEGNET: An Input-Agnostic Self-Supervised EEG Rrpresentation Model For Inter-Dataset Transfer Learning



---

**Abstract:** 
There is currently a scarcity of labeled Electroencephalog-
raphy (EEG) recordings, and different datasets usually have
incompatible setups (e.g., various sampling rates, number of
channels, event lengths, etc.). These issues hinder machine
learning practitioners from training general-purpose EEG
models that can be reused on specific EEG classification
tasks through transfer learning. We present a deep convo-
lutional neural network architecture with a spatial pyramid
pooling layer that is able to take in EEG signals of varying
dimensionality and extract their features to fixed-size vec-
tors. The model is trained with a contrastive self-supervised
learning task on a large unlabelled dataset. We introduce a
set of EEG signal augmentation techniques to generate large
amounts of sample pairs to train the feature extractor. We
then transfer the trained feature extractor to new downstream
tasks. The experimental results show that the first few con-
volutional layers of the feature extractor learn the general
features of the EEG data, which can significantly improve the
classification performance on new datasets.




---

**Methodology:**



1.   SPP-EEG Feature extractor
2.   Contrastive Learning
3.   Transfer learning

---


**Code Structure:**

-datasets/

-models/

-preprocessing/

-downstream/

-ssl_train.py




