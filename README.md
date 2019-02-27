# Netkov_gh

Active and messy. Currently training.. Cleaning and More to come here soon. 

Inception and ResNet modules are included so that the Normalization technique can be changed depending on the training procedure (i.e. proxy or triplet). If you are not using transfer learning, changing Batch Normalization to Group normalization gives a more efficient and stable training procedure, especially if you have small batch size, for Triplet sampling and learning due to loss of samples being consistent during both eval and train modes.

train.py contains codes to use triplet learning.    

Netkov.py contains codes to use different resnet, inception architectures with or without parallel shallow networks (as in below papers, 
not suitable for Transfer Learning).  

ProxyUtils.py is for implementing proxy loss, proxy learning testing and data loading. Modify it according to your needs.  

ProxyTrain.py is for using proxy learning proposed in the first paper below.  

Works used in/inspired this repo:  
"No Fuss Distance Metric Learning using Proxies" by Yair Movshovitz-Attias, Alexander Toshev, Thomas K. Leung, Sergey Ioffe, Saurabh Singh (arXiv: 1703.07464)

"Learning Fine-grained Image Similarity with Deep Ranking" by Jiang Wang, Yang song, Thomas Leung, Chuck Rosenberg, Jinbin Wang, James Philbin, Bo Chen, Ying Wu (arXiv:1404.4661v1)

"Where to Buy It: Matching Street Clothing Photos in Online Shops" by M. Hadi Kiapour, Xufeng Han, Svetlana Lazebnik, Alexander C. Berg, Tamara L. Berg

"Deep Learning based Large Scale Visual Recommendation and Search for E-Commerce" by Devashish Shankar, Sujay Narumanchi, H A Ananya, Pramod Kompalli, Krishnendu Chaudhury (arXiv:1703.02344)

"Group Normalization" by Yuxin Wu, Kaiming He (arXiv:1803.08494)
