SAR Image Despeckling with Gradient-Guided by Cross-domain Reference Feature derived from optical and SAR images
Yang Yang a, Jun Pan a,*, Jiangong Xu a, Zhongli Fan a, Zeming Geng a, Junli Li b, Mi Wang a,*
a State Key Laboratory of Information Engineering in Surveying, Mapping and Remote Sensing, Wuhan University, Wuhan 430079, China
b National Key Laboratory of Ecological Security and Sustainable Development in Arid Region, Xinjiang Institute of Ecology and Geography, Chinese Academy of Sciences, Urumqi 830011, China

![image](https://github.com/user-attachments/assets/151e8a5d-ce42-4028-adfc-4e6641f7c9df)

![image](https://github.com/user-attachments/assets/bc0fe0a3-8f5a-4456-8289-c57f8e224bd6)
Despecking results of different methods on simulated images

![image](https://github.com/user-attachments/assets/1d8b3b03-995c-4858-ae51-2e7ff623985e)
Despecking results of different methods on real images

The code needs to run under ubuntu 20.04 with opencv 3.4 installed
1.Run the following code to install the environment

conda create -n MBCR python==3.8
pip install -r requirements.txt

2.Cross-domain reference map generation
Go to the gen_ref folder, change the path to the dataset in nearestPixel.py, and run python nearestPixel.py to generate the reference image NLsar

3.train.  Put the data together in the format of the example in the dataset file, and then run

python  train.py --session_name    <new_session>  --epoch  20 

4.test Modify the path to the data in test and run

python test.py --session_name  <new_session> --ckpt_epoch 20
