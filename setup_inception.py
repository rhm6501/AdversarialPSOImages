import os.path
import numpy as np
import tensorflow as tf
import PIL

def readimg(ff, path,labs):
  f = os.path.join(path, ff)
  img = load_image(f)
  return [img, labs[ff], f]

def load_image(path):
    image = PIL.Image.open(path)
    if image.height > image.width:
        height_off = int((image.height - image.width)/2)
        image = image.crop((0, height_off, image.width, height_off+image.width))
    elif image.width > image.height:
        width_off = int((image.width - image.height)/2)
        image = image.crop((width_off, 0, width_off+image.height, image.height))
    image = image.resize((299, 299))
    img = np.asarray(image).astype(np.float32) / 255.0
    if img.ndim == 2:
        img = np.repeat(img[:,:,np.newaxis], repeats=3, axis=2)
    if img.shape[2] == 4:
        img = img[:,:,:3]
    return img
  
  
class ImageNet:
  def __init__(self, path=os.path.join('.','imagenet','images'),indices=None):
    labs={}
    with open('val.txt','r') as f:
        for lines in f:
            a,b=lines.split(" ")
            labs[a]=int(b.strip())
    
    read_img_fn = lambda x: readimg(x, path,labs)
    file_list = sorted(os.listdir(path))
    if len(indices)==0:
        indices=1000
        r = [read_img_fn(x) for x in  file_list[:indices]]
    else:
        r = [read_img_fn(file_list[i]) for i in indices]
    r = [x for x in r if x != None]
    test_data, test_labels, test_paths = zip(*r)
    self.test_data = np.array(test_data)
    self.test_labels = np.zeros((len(test_labels), 1000))
    self.test_labels[np.arange(len(test_labels)), test_labels] = 1
    self.test_paths = test_paths

if __name__ == '__main__':
  tf.app.run()
