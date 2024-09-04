## SAR Image Despeckling with Gradient-Guided by Cross-domain Reference Feature derived from optical and SAR images
Yang Yang<sup>a</sup>, Jun Pan <sup>a,※</sup>, Jiangong Xu <sup>a</sup>, Zhongli Fan <sup>a</sup>, Zeming Geng <sup>a</sup>, Junli Li <sup>b</sup>, Mi Wang <sup>a,※</sup>

<sup>a</sup> State Key Laboratory of Information Engineering in Surveying, Mapping and Remote Sensing, Wuhan University, Wuhan 430079, China

<sup>b</sup> National Key Laboratory of Ecological Security and Sustainable Development in Arid Region, Xinjiang Institute of Ecology and Geography, Chinese Academy of Sciences, Urumqi 830011, China

![image](https://github.com/user-attachments/assets/151e8a5d-ce42-4028-adfc-4e6641f7c9df)

![image](https://github.com/user-attachments/assets/bc0fe0a3-8f5a-4456-8289-c57f8e224bd6)
Despecking results of different methods on simulated images

![image](https://github.com/user-attachments/assets/1d8b3b03-995c-4858-ae51-2e7ff623985e)
Despecking results of different methods on real images

## The dataset are as follows:
QXS-SAROPT: https://www.wenjuan.com/s/UZBZJv5GwL/
Hunan_dataset: https://drive.google.com/file/d/1m3wYiQolm2YEmpzH6cGQZ4wWw_EdFAdN/view?usp=sharing
Xinjiang_dataset: https://drive.google.com/drive/folders/1uSLrDiFrS-3ydES5jaiccnEKz0loVwu1?usp=drive_link


The code needs to run under ubuntu 20.04 with opencv 3.4 installed
## 1.Run the following code to install the environment

`conda create -n MBCR python==3.8`
`pip install -r requirements.txt`

## 2.Cross-domain reference map generation
1. run `g++ -o nearestPixel nearestPixel.cpp `pkg-config --cflags --libs opencv` to generate .so file 
2. Go to the gen_ref folder, change the path to the dataset in nearestPixel.py, and run `python nearestPixel.py` to generate the reference image NLsar

## 3.train.  Put the data together in the format of the example in the dataset file, and then run

`python  train.py --session_name    <new_session>  --epoch  20 `

## 4.test Modify the path to the data in test and run

`python test.py --session_name  <new_session> --ckpt_epoch 20`

