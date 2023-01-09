################################
#
################################

import cv2
import matplotlib.pyplot as plt
import numpy as np

DATA_PATH="data/sample.png"
def sample():
    print("sample start ")
    version = cv2.__version__

    img = cv2.imread(DATA_PATH)
    print(version)
    print(type(img))
    print(img.shape)
    print(img.dtype)

    print(img)

    plt.imshow(img)
    plt.savefig('data/opencv_img.jpg')
    plt.close()

    # BGR -> RGB
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.savefig('data/opencv_rgb.jpg')
    plt.close()



def pillow():
    print("pilllow start ")
    """
    Pillow
    """
    from PIL import Image

    img = Image.open( DATA_PATH)
    print(type(img))
    """ not Numpy class ,, convert to numpy """
    import numpy as np
    img = np.array( img)
    print(type(img))
    print(img.shape)

    """ convert gray scale"""
    import cv2
    import matplotlib.pyplot as plt
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#    plt.imshow(img_gray)
    print(img_gray.shape)

    plt.gray()
    plt.imshow(img_gray)
    print(img_gray.shape)

    plt.savefig('data/opencv_gray_cvtcolor.jpg')
    plt.close()
    return img_gray

def cnn( img_gray ):
    print("cnn start")
    kernel = np.array([
            [-1,0,1],
            [-1,0,1],
            [-1,0,1],
            ])
    img_conv = cv2.filter2D( img_gray,-1,kernel ) 
    plt.gray()
    plt.imshow(img_conv)
    plt.savefig('data/opencv_gray_filtered_vertical.jpg')
    plt.close()

    kernel = np.array([
            [-1,-1,-1],
            [0,0,0],
            [1,1,1],
            ])
    img_conv = cv2.filter2D( img_gray,-1,kernel ) 
    plt.gray()
    plt.imshow(img_conv)
    plt.savefig('data/opencv_gray_filtered_horizontal.jpg')
    plt.close()

    print("laplacian filter")
    kernel = np.array([
            [1,1,1],
            [1,-8,1],
            [1,1,1],
            ])
    img_conv = cv2.filter2D( img_gray,-1,kernel ) 
    plt.gray()
    plt.imshow(img_conv)
    plt.savefig('data/opencv_gray_filtered_laplacian.jpg')
    plt.close()

    print("smooth filter")
    kernel = np.array([
            [1,1,1],
            [1,1,1],
            [1,1,1],
            ])/9
    img_conv = cv2.filter2D( img_gray,-1,kernel ) 
    plt.gray()
    plt.imshow(img_conv)
    plt.savefig('data/opencv_gray_filtered_smooth.jpg')
    plt.close()



def main():
    sample()
    img_gray = pillow()
    cnn(img_gray)

main()
