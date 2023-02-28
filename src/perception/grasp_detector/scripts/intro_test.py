import scipy
import cv2
import numpy as np

# data_dir = '/home/teamf/rec_data/workspace_mask.png'
# img  = np.zeros((480,640))

# # img_color  =cv2.imread('/home/teamf/rec_data/color_0.png')
# # cv2.imshow('image',img_color)
# # cv2.waitKey(10)

# # img[91:-1,102:235] = 255
# img[:,:] = 255


# cv2.imwrite(data_dir,img)


# mat = scipy.io.loadmat('/home/teamf/graspnet/graspnet-baseline/doc/example_data/meta.mat')

# #setting intrinsic params
# # mat["intrinsic_matrix"] = np.array([[614.8798828125, 0.0, 318.8515625],[ 0.0, 614.9976806640625, 248.05238342285156],[ 0.0, 0.0, 1.0]])
# # mat["factor_depth"] = 1000

# # 
# # mat["intrinsic_matrix"] = np.array([[0.501, 0.801, 0.5],[ 0.506, -0.056, 0.065],[ -0.001, -0.001, -0.021]])
# # mat["factor_depth"] = 0.001

# scipy.io.savemat('/home/teamf/rec_data/meta.mat',mat)

# print(mat)







#to get mask coordinates
def draw_circle(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
im1 = cv2.imread('/home/teamf/rec_data/frame000001.png')

cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)
cv2.imshow('image', im1)
cv2.waitKey(0)