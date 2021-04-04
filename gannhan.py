from imutils import paths
import cv2
import pickle
import numpy as np
DATASET_PATH = "./Database"

def _model_processing():

    image_links = list(paths.list_images(DATASET_PATH))# list tat ca cac anh trong thu muc database.
    #images_file = [] 
    y_labels = []# tao 1 ma tran gan nhan
    faces = []# ma tran khuon mat.

    for image_link in image_links:
        split_img_links = image_link.split("\\") # tro den duong link cua file anh moi nguoi co trong database.
    # Lấy nhãn của ảnh
        name = split_img_links[-2] # dat ten bang ten cua duong link thu muc.
    # Đọc ảnh
        face = cv2.imread(image_link) # doc anh 
        faces.append(face) # luu het tat ca chi so ma tran ten anh vao face []
        y_labels.append(name) #luu tat ca y_labels(ten) vao ma tran 
        #=> faces[]===y_labels

     #   images_file.append(image_links)
    return faces, y_labels, images_file

faces, y_labels, images_file = _model_processing()
faces=np.asarray(faces)
y_labels=np.asarray(y_labels)

print(faces.shape)
print(y_labels.shape)
def _save_pickle(obj, file_path): #ham luu 
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)

def _load_pickle(file_path): #ham doc
    with open(file_path, 'rb') as f:
        obj = pickle.load(f)
    return obj

_save_pickle(faces, "./faces.pkl") # khi qua trich dac trung chi can goi lai ham nay de doc
_save_pickle(y_labels, "./y_labels.pkl")
_save_pickle(images_file, "./images_file.pkl")