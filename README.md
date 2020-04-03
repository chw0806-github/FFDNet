# FFDNet for SAR image despeckling
Repository for the project of the class 'Remote sensing data'. 


Based upon the paper: "Zhang, K., Zuo, W., & Zhang, L. (2018). FFDNet: Toward a fast and flexible solution for CNN-based image denoising" [link](https://arxiv.org/abs/1710.04026).

How to run the project:

The project was developed and tested mainly using Google Colab to have GPU support.

1. Copy the project to a Google Drive
2. Open code/main_notebook.ipynb with Google Colab

Where to find visual results:

1. data/Image_Database contains the trained network outputs of the Test images (artificial speckle)
2. data/Real_images_denoised contains despeckled versions of Real_images (real SAR images)
3. data/project_FFDNet_eval contains input and output of an evaluation set, that is comparable to other methods.


To keep this repo in a reasonable size it only contains a portion of our training/evaluation data. The complete data can be found  [here](https://drive.google.com/open?id=1_BNv4Pj6rbZ3jYoTpGV0h9Z5enBRO6Ik).
