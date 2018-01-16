
import numpy as np
import pandas as pd
import argparse

def preprocessing(filePath):
    from keras.preprocessing import image as image_utils
    ## get class number to labels
    pathDict = pd.read_csv(filePath, header = None).as_matrix()
    classDict = {}
    for index, c in enumerate(pathDict):
        classDict[c[0] ]=index
    #print(classDict)
    test_data = np.zeros((10000, 64, 64, 3))
    ## get labels from val_annotations.txt
    test_class = pd.read_csv("./val/val_annotations.txt", header = None, sep = "\t")
    test_label = np.zeros((10000,))
    for index, className in enumerate(test_class[1]):
        test_label[index] = classDict[className]
    #print(test_label)
    test_data = np.zeros((10000, 64, 64, 3))
    for index in range(10000):
        imgPath = './val/images/val_' + str(index) + '.JPEG'
        image = image_utils.load_img(imgPath, target_size = (64, 64))
        image = image_utils.img_to_array(image)
        test_data[index] = image
        ### for test running
        #train_data[index * 5 + imgNum] = image
    
    print('Read Done!!')
    return test_data, test_label 

def main():
    ### data preprocessing
    parser = argparse.ArgumentParser(description = "MobileNet on Tiny ImageNet Dataset")
    # Retrievement argument
    parser.add_argument('-f', '--file', type = str, help = 'FileDict Name Input', default = "wnids.txt")
    parser.add_argument('-m', '--model', type = str, help = 'Model name', default = "./models/AlexNet.h5")
    args = parser.parse_args()


    ## testing data preprocessing
    test_data, test_label = preprocessing(args.file)
    #print(test_data[-1])
    from keras.utils import to_categorical
    test_label = to_categorical(test_label, num_classes = 200)

    ## model   
    from keras.models import load_model
    model = load_model(args.model)

    ## testing
    result = model.evaluate(test_data, test_label)
    print(result)


if __name__ == '__main__':
    main()  