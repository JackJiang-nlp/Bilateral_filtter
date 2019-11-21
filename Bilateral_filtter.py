
"""
在运行代码前查看readme.txt文件，查看程序运行的环境
"""
# %%
import cv2
import numpy as np
import torch
import time
import send_mail
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# %%
PI = 3.14
class Filter:
    def __init__(self, image_path, output_path, r, sigma_d, sigma_r):
        self.image = cv2.imread(image_path)
        self.change_image=cv2.imread(image_path)
        self.nchannels=None
        num_channels=len(self.image.shape)
        if num_channels==3:
            self.rows, self.cols, self.nchannels = self.image.shape
        else :
            self.rows, self.cols = self.image.shape
        self.output_path = output_path
        self.r = r
        self.w_filter = 2*r+1  # size of the filter
        self.sigmaD_s = -2*sigma_d**2
        self.sigmaR_s = -2*sigma_r**2

    def position_ker(self):
        col = row = torch.Tensor(np.linspace(-self.r, self.r, self.w_filter) *
                                 np.linspace(-self.r, self.r, self.w_filter))
        if self.nchannels is None:
            W_col = col.unsqueeze(1).expand(self.w_filter, self.w_filter)
            W_row = row.unsqueeze(0).expand(self.w_filter, self.w_filter)
        else:
            W_col = col.unsqueeze(1).unsqueeze(2).expand(
                self.w_filter, self.w_filter, self.nchannels)
            # print(W_col)
            W_row = col.unsqueeze(0).unsqueeze(2).expand(
                self.w_filter, self.w_filter, self.nchannels)
            # print(W_row)
        # Pker=torch.exp((W_col+W_row)/self.sigmaD_s)
        self.Sker = ((W_col+W_row)/self.sigmaD_s).numpy()
        # return self.Sker

    def value_ker(self, index_x, index_y):
        if self.nchannels is None:
            image_ker = self.image[index_x-self.r:index_x+self.r+1,
                                index_y-self.r:index_y+self.r+1]
            diff_ker = np.array(image_ker,dtype=np.int32) - \
                self.image[index_x, index_y]      
            Dker_s = diff_ker**2/self.sigmaR_s
        else:
            image_ker = self.image[index_x-self.r:index_x+self.r+1,
                                index_y-self.r:index_y+self.r+1,:]
            diff_ker = image_ker - \
                np.array(self.image[index_x, index_y, :],
                        dtype=np.int32)
            Dker_s = diff_ker**2/self.sigmaR_s
        return Dker_s

    def bilateral_filtter(self, index_x, index_y):
        Dker_s = self.value_ker(index_x=index_x, index_y=index_y)
        final_ker = np.exp(Dker_s+self.Sker)
        if self.nchannels is None:
            self.change_image[index_x, index_y] = (self.image[index_x-self.r:index_x+self.r+1,
                                        index_y-self.r:index_y+self.r+1]*final_ker).sum(
                                      axis=0).sum(axis=0)/(final_ker.sum(axis=0).sum(axis=0))
        else:
            self.change_image[index_x, index_y, :] = (self.image[index_x-self.r:index_x+self.r+1,
                                        index_y-self.r:index_y+self.r+1, :]*final_ker).sum(
                                      axis=0).sum(axis=0)/(final_ker.sum(axis=0).sum(axis=0))

    def Conv_fun(self):
        self.position_ker()
        for i in range(self.r, self.rows-self.r):
            for j in range(self.r, self.cols-self.r):
                self.bilateral_filtter(i, j)
        cv2.imwrite(self.output_path, self.change_image)
        return self.image

# %%

# def Normalize(data):
#     max = np.max(data[:, :, 0])
#     min = np.min(data[:, :, 0])
#     diff = max-min
#     mean = np.mean(data[:, :, 1])
#     norm_data = (data-mean)/diff
#     return norm_data

# %%

# def plot(data, r):
#     fig = plt.figure()
#     ax = plt.axes(projection='3d')
#     x_line = np.linspace(0, r/2, 2*r+1)
#     y_line = np.linspace(0, r/2, 2*r+1)
#     x_line, y_line = np.meshgrid(x_line, y_line)
#     z_line = data[:, :, 1]
#     ax.plot_surface(x_line, y_line, z_line)
#     plt.show()

# def callback(object):
#     pass


# %%
if __name__ == "__main__":
    run_time = time.time()
    image_path = "./images/Circuit_noise.jpg"
    half_width = 2
    sigmaD = 200
    sigmaR = 80
    output_path = "./output1_image/{}_{}_{}.jpg".format(half_width,sigmaD,sigmaR)
    filter = Filter(image_path, output_path, half_width, sigmaD, sigmaR)
    out_img = filter.Conv_fun()
    # cv2.imshow('image',out_img)
    # cv2.waitKey(0)
    # for half_width in range(1,3): 
    #     for sigmaD in np.linspace(310,600,30):
    #         for sigmaR in np.linspace(110,250,15):  
    #             output_path = "./output_image/{}_{}_{}.jpg".format(half_width,sigmaD,sigmaR)
    #             filter = Filter(image_path, output_path, half_width, sigmaD, sigmaR)
    #             out_img = filter.Conv_fun()
    # end_time=time.time()
    # print(end_time-run_time)
    # send_mail.Smail() 


    # cv2.namedWindow("image")
    # cv2.createTrackbar("d", "image", 0, 255, callback)
    # cv2.createTrackbar("sigmaColor", "image", 0, 255, callback)
    # cv2.createTrackbar("sigmaSpace", "image", 0, 255, callback)
    # while(1):
    #     half_width = cv2.getTrackbarPos("d", "image")
    #     sigmaD= cv2.getTrackbarPos("sigmaSpace", "image")
    #     sigmaR=cv2.getTrackbarPos("sigmaColor", "image")
    #     image_path="./images/Circuit_noise.jpg"

    #     output_path="./output_image/{}_{}.jpg".format(int(10.9),int(10.9))

    #     filter=Filter(image_path,output_path,half_width,sigmaD,sigmaR)
    #     out_img=filter.Conv_fun()
    #     cv2.imshow("out", out_img)
    #     k = cv2.waitKey(1)
    #     if k == 27:
    #         break
    # cv2.destroyAllWindows()
    # 测试
    # ker_sum=filter.position_ker()
    # gaussian=np.exp(ker_sum)/(2*PI*(sigmaD**2))
    # # gaussian1=Normalize(gaussian)
    # plot(gaussian,half_width)
    # print("successful")
