#coding = gbk
from keras.models import load_model
import numpy as np
import scipy.signal as sc
import sys
import math
import gc
#import matplotlib.pyplot as plt


class HolterSystem_Signal(object):
    """ECG信号的读取与预测"""
    def __init__(self):
        self.pwd = sys.argv[0]
        #rlt文件地址
        self.path = self.pwd.replace("rlt.py", "dir.txt")
        # model地址
        self.cnn_model_path = self.pwd.replace("rlt.py", "cnn_model_1.h5")

    def read_rlt_ecg_file_path(self):
        """产生rlt文件位置，ecg文件位置，第几导联 """
        file = open(self.path, "r",encoding='utf-8').readlines()
        rlt_path = file[0].strip("\n")
        ecg_path = file[1].strip("\n")
        NUM = file[2].strip("\n")
        return rlt_path, ecg_path, NUM

    def read_r_point(self, rlt_path):
        """读取r_point"""
        rlt_file = open(rlt_path, 'rb+').read()
        rlt_head = rlt_file[:256]
        rlt_data = rlt_file[256:]
        r_point=[]
        for i in range(rlt_head[4] + rlt_head[5] * 256+rlt_head[6] * 256**2 + rlt_head[7] * 256**3):
            r_point.append(rlt_data[i*38] + rlt_data[1+i*38] * 256 + rlt_data[2+i*38] * 256**2+rlt_data[3+i*38] * 256**3)
        return r_point

    def read_ecg(self, ecg_path, NUM):
        """读取ecg文件，NUM代表第几导联"""
        ecg_file = open(ecg_path, "rb").read()
        ecg_file = ecg_file[64:]
        data_len = int(len(ecg_file) / 8)
        if NUM =="0": #I
            num=0
        if NUM =="1": #II
            num=1
        if NUM =="6":#V1
            num = 2
        if NUM =="7":#V2
            num = 3
        if NUM =="8":#V3
            num = 4
        if NUM =="9":#V4
            num = 5
        if NUM =="10":#V5
            num = 6
        if NUM =="11":#V6
            num = 7
        ecg_signal = [ecg_file[num+8*i] for i in range(data_len)]
        return ecg_signal

    def ecg_filter(self,ecg_signal):
        """对信号进行过滤 """
        b, a = sc.butter(3, (0.016, 0.48), 'bandpass')
        signal_bandpass = sc.filtfilt(b, a, ecg_signal)
        return signal_bandpass

    def split_ecg(self, ecg_signal, r_point):
        """对信号进行切分"""
        split_ecg = []
        for point in r_point:
            if point > 34:
                split_ecg.append(ecg_signal[point - 34:point + 62])
            else:
                split_ecg.append(ecg_signal[:96])
        return split_ecg

    def signal_normalization(self,split_ecg):
        """对切分后的信号进行归一化"""
        nor_split_ecg = []
        for ecg in split_ecg:
            max_0 = max(ecg)
            min_0 = min(ecg)
            N = max_0 - min_0 + 0.0001
            nor_split_ecg.append([(i - min_0) / N for i in ecg])
        return nor_split_ecg

    def prediction_results(self,nor_split_ecg):
        """对处理完毕的信号进行预测"""
        cnn_model = load_model(self.cnn_model_path)
        nor_split_ecg = np.array(nor_split_ecg).reshape(len(nor_split_ecg), 96, 1)
        lable = cnn_model.predict(nor_split_ecg, batch_size=128, verbose=0)
        return lable


class HolterSystem_Noise(object):
    """对信号进行噪声分析"""
    def __init__(self):
        # rlt文件地址
        self.pwd = sys.argv[0]
        #正常信号的model
        self.zhengchang_model = np.load(self.pwd.replace("rlt.py", "zhengchang_model.npy"))
        #多源信号的model
        self.duoyuan_model = np.load(self.pwd.replace("rlt.py", "duoyuan_model.npy"))
        #单源信号的model
        self.danyuan_model = np.load(self.pwd.replace("rlt.py", "danyuan_model.npy"))
        #自编码模型
        self.gan_path = self.pwd.replace("rlt.py", "gan_model_30.h5")


    def consine_dis(self,x, y):
        """比较两个信号的相似度"""
        dot = sum([a * b for a, b in zip(x, y)])
        denom = np.linalg.norm(x) * np.linalg.norm(y)
        result = dot / denom
        return result

    def signal_step(self,x):
        """计算信号的阶跃"""
        num = 0
        for i in range(20):
            if max(x[i*2:(i+1)*2])-min(x[i*2:(i+1)*2])>=0.07:
                num+=1
        return num

    def noise_level(self,original_ecg):
        """噪声水平"""
        MEAN = sum(original_ecg) / 96.0
        SDNN = math.sqrt(sum([(i - MEAN) ** 2 for i in original_ecg]) / 96.0)
        return SDNN/(MEAN+0.00001)

    def peak(self,x):
        """峰值裕度"""
        x_mean = np.mean(x)
        a = sum([(i - x_mean) ** 4 for i in x]) / len(x)
        b = (sum([(i - x_mean) ** 2 for i in x]) / len(x)) ** 2
        if b == 0:
            return 0
        else:
            return a / b

    def is_noise(self,original_ecg,A,MAX,MIN,M_128):#单源0.35 多源0.4
        """噪声判断"""
        flag = 0
        level = self.noise_level(original_ecg)
        extremum_max = len([item for item in original_ecg if item == 255])#有的多源室早幅度很大 3月20
        extremum_min = len([item for item in original_ecg if item == 0])
        extremum_128 = len([item for item in original_ecg if item == 128])
        if level>A or extremum_max>MAX or extremum_min>MIN or extremum_128>M_128:
            flag = 1
        return flag


    def ecg_step(self,ecg_nor):
        #寻找信号中不合适的拐弯点
        sum = 0
        for j in range(len(ecg_nor[28:43])):
            if ecg_nor[28 + j] < ecg_nor[28 + j - 1] and ecg_nor[28 + j] < \
                    ecg_nor[28 + j + 1]:
                if ecg_nor[34] > 0.6 and ecg_nor[34] - ecg_nor[28 + j] > 0.3 and ecg_nor[28 + j] > 0.2:
                    if 28 + j < 33:
                        sum+=1

        return sum

    def rewrite_ecg_lable(self,rlt_path,ecg_lable,original_ecg,ecg_nor):
        file = open(rlt_path, 'rb+')
        rlt = bytearray(file.read())
        file.close()
        rlt_title = rlt[:256]
        rlt_data = rlt[256:]
        gan_model = load_model(self.gan_path)
        cnn_model = load_model(HolterSystem_Signal().cnn_model_path)
        for i in range(1,len(ecg_lable)-1):
            #if (rlt_data[4+i*38]+rlt_data[5+i * 38]*256)<1200:
                Is_Noise = self.is_noise(original_ecg[i],0.4,5,4,48)
                if Is_Noise:
                    rlt_data[4 + i * 38] = 176
                    rlt_data[5 + i * 38] = 4
                else:
                    #首先判断是不是噪声
                    if ecg_lable[i]==0: #如果是单源
                        peaks, _ = sc.find_peaks(ecg_nor[i], distance=3, prominence=(0.1, 5))
                        gan_ecg = gan_model.predict(np.array(ecg_nor[i]).reshape(1, 96))
                        sim = sum([abs(a - b) for a, b in zip(gan_ecg[0], ecg_nor[i])])
                        COS_dan = self.consine_dis(gan_ecg[0], self.danyuan_model)
                        if self.peak(original_ecg[i])>11.2:
                            rlt_data[4 + i * 38] = 0
                            rlt_data[5 + i * 38] = 0
                        else:
                            if sim < 6.5 and len(peaks) <= 3:
                                if self.peak(original_ecg[i]) > 3.3 or COS_dan > 0.98:
                                    rlt_data[4 + i * 38] = 200
                                    rlt_data[5 + i * 38] = 0
                                else:
                                    rlt_data[4 + i * 38] = 176
                                    rlt_data[5 + i * 38] = 4
                            else:
                                rlt_data[4 + i * 38] = 176
                                rlt_data[5 + i * 38] = 4

                    if ecg_lable[i]==2: #如果是多源
                           peaks, _ = sc.find_peaks(ecg_nor[i], distance=3, prominence=(0.1, 5))
                           gan_ecg = gan_model.predict(np.array(ecg_nor[i]).reshape(1, 96))
                           sim = sum([abs(a - b) for a, b in zip(gan_ecg[0], ecg_nor[i])])
                           pea = self.peak(ecg_nor[i])
                           if  pea>4 and sim<8 and len(peaks)<=3 and self.ecg_step(ecg_nor[i])==0:
                                rlt_data[4 + i * 38] = 200
                                rlt_data[5 + i * 38] = 0
                           else:
                                rlt_data[4 + i * 38] = 176
                                rlt_data[5 + i * 38] = 4

                    if ecg_lable[i]==3:
                        pea = self.peak(ecg_nor[i])
                        gan_ecg = gan_model.predict(np.array(ecg_nor[i]).reshape(1, 96))
                        sim = sum([abs(a - b) for a, b in zip(gan_ecg[0], ecg_nor[i])])
                        peaks, _ = sc.find_peaks(ecg_nor[i], distance=3, prominence=(0.1, 5))
                        if pea>2 and sim<13 and len(peaks)<=4:
                            gan_lable = np.argmax(cnn_model.predict(gan_ecg[0].reshape(1, 96, 1)))
                            if gan_lable==3:
                                rlt_data[4 + i * 38] = 200
                                rlt_data[5 + i * 38] = 0
                        else:
                            rlt_data[4 + i * 38] = 176
                            rlt_data[5 + i * 38] = 4

        for i in range(2,len(ecg_lable)-2):
            if (ecg_lable[i-1]==3 or ecg_lable[i-1]==0) and ecg_lable[i]==0:
                if rlt_data[4 + (i-1) * 38] == 200:
                    peaks, _ = sc.find_peaks(ecg_nor[i], distance=3, prominence=(0.1, 5))
                    gan_ecg = gan_model.predict(np.array(ecg_nor[i]).reshape(1, 96))
                    sim = sum([abs(a - b) for a, b in zip(gan_ecg[0], ecg_nor[i])])
                    if len(peaks)<=3 and sim<13:
                        rlt_data[4 + i * 38] = 200
                        rlt_data[5 + i * 38] = 0


        for i in range(2,len(ecg_lable)-2):
            if rlt_data[4 + (i-1) * 38]==176 and rlt_data[4 + (i-2) * 38]==176 and rlt_data[4 + (i+1) * 38]==176 and rlt_data[4 + (i+2) * 38]==176:
                    rlt_data[4 + i * 38] = 176
                    rlt_data[5 + i * 38] = 4

        for i in range(1,len(ecg_lable)-1):
            if (ecg_lable[i] == 2 or ecg_lable[i]==0) and r_point[i]-r_point[i-1]>50 and r_point[i+1]-r_point[i]>50:
                gan_ecg = gan_model.predict(np.array(ecg_nor[i]).reshape(1, 96))
                sim = sum([abs(a - b) for a, b in zip(gan_ecg[0], ecg_nor[i])])
                if sim>13:
                    rlt_data[4 + i * 38] = 176
                    rlt_data[5 + i * 38] = 4

        for i in range(1, len(ecg_lable) - 1):
            if (ecg_lable[i] == 2 or ecg_lable[i] == 0) and rlt_data[4 + i * 38] == 176:
                gan_ecg = gan_model.predict(np.array(ecg_nor[i]).reshape(1, 96))
                sim = sum([abs(a - b) for a, b in zip(gan_ecg[0], ecg_nor[i])])
                if self.ecg_step(ecg_nor[i])==0 :
                    if ecg_lable[i]==2  and sim<4.3 and self.is_noise(original_ecg[i],0.45,0,8,48)==0:
                        rlt_data[4 + i * 38] = 200
                        rlt_data[5 + i * 38] = 0
                    if ecg_lable[i]==0 and sim<4 and self.is_noise(original_ecg[i],0.4,5,4,48)==0:
                        rlt_data[4 + i * 38] = 200
                        rlt_data[5 + i * 38] = 0

            if ecg_lable[i] == 1 and rlt_data[4 + i * 38] == 176:
                gan_ecg = gan_model.predict(np.array(ecg_nor[i]).reshape(1, 96))
                sim = sum([abs(a - b) for a, b in zip(gan_ecg[0], ecg_nor[i])])
                if sim<6:
                    rlt_data[4 + i * 38] = 0
                    rlt_data[5 + i * 38] = 0


        for i in range(2,len(ecg_lable)-2):
            if rlt_data[4 + i * 38]==200:
                s = 0
                for j in range(-2,3):
                    if rlt_data[4 + (i+j) * 38]==176:
                        s+=1
                if s>=3 and ecg_lable[i]!=3:
                    rlt_data[4 + i * 38] = 176
                    rlt_data[5 + i * 38] = 4

        file = open(rlt_path, 'wb')
        file.write(bytearray(rlt_title + rlt_data))
        file.close()



if __name__ =='__main__':
    rlt_path, ecg_path, NUM = HolterSystem_Signal().read_rlt_ecg_file_path()
    #读取R点
    r_point= HolterSystem_Signal().read_r_point(rlt_path)
    #读取ecg
    ecg_signal = HolterSystem_Signal().read_ecg(ecg_path,NUM)
    #切分原始信号
    split_original_ecg = HolterSystem_Signal().split_ecg(ecg_signal,r_point)
    #滤后信号
    ecg_filte = HolterSystem_Signal().ecg_filter(ecg_signal)

    #切分过滤后信号
    split_bandpass_ecg = HolterSystem_Signal().split_ecg(ecg_filte,r_point)

    #归一化后信号
    ecg_nor = HolterSystem_Signal().signal_normalization(split_bandpass_ecg)
    #预测
    ecg_lable = HolterSystem_Signal().prediction_results(ecg_nor)

    lable = []
    for i in range(len(ecg_lable)):
        if np.argmax(ecg_lable[i]) == 0:  # 单源
            lable.append(0)
        if np.argmax(ecg_lable[i]) == 1:  # 正常
            lable.append(1)
        if np.argmax(ecg_lable[i]) == 2:  # 多源
            lable.append(2)
        # if np.argmax(ecg_lable[i]) == 3:  # 深S
        #     lable.append(1)
        if np.argmax(ecg_lable[i]) == 3:#连续单源
            lable.append(3)
    del ecg_lable
    gc.collect()
    #改写rlt文件
    HolterSystem_Noise().rewrite_ecg_lable(rlt_path,lable,split_original_ecg,ecg_nor)
    #