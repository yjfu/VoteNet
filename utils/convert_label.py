import os
from PIL import Image
import matplotlib.image as mpimg
import scipy.io
import numpy as np
import shutil
work_path = os.path.join("/", "home", "smartcar", "fuyj")
VOC_ori_index_path = os.path.join(work_path, "VoteNet", "data",
                           "voc_2012_valid.txt")

AUG_ori_index_path = os.path.join(work_path, "VoteNet", "data",
                           "voc_aug_valid.txt")

merged_index_path = os.path.join(work_path, "VoteNet", "data",
                           "val.txt")
mask_out_dir = os.path.join(work_path, "VOC_Seg", "mask")


label_colours = [(0,0,0)
                # 0=background
                ,(128,0,0),(0,128,0),(128,128,0),(0,0,128),(128,0,128)
                # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                ,(0,128,128),(128,128,128),(64,0,0),(192,0,0),(64,128,0)
                # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                ,(192,128,0),(64,0,128),(192,0,128),(64,128,128),(192,128,128)
                # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
                ,(0,64,0),(128,64,0),(0,192,0),(128,192,0),(0,64,128)
                # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                ]

def convert_VOC_2012(work_path, VOC_ori_index_path, VOC_converted_mask_dir):
    """
    convert VOC mask file from rgb to id, masked with id of 1,2...20,
    and 255 for ignored.
    :param work_path: base path of index file
    :param VOC_ori_index_path: file of 'path/to/img path/to/mask'
    :param VOC_converted_mask_dir: new mask files will locate here
    :return:
    """
    lines = open(VOC_ori_index_path, "r").readlines()
    for line in lines:
        mask_path = os.path.join(work_path, line.split()[1])
        # here using PIL.Image will automatically change RGB to 0~20 using P mode
        # and 255 for ignored pixels
        mask = Image.open(mask_path)
        mask = np.array(mask)

        mask_file_name = mask_path.split('/')[-1]
        out_path = os.path.join(VOC_converted_mask_dir, mask_file_name)

        mask = Image.fromarray(mask, mode="L")
        mask.save(out_path)

def convert_VOC_Aug(work_path, AUG_ori_index_path, AUG_converted_mask_dir):
    """
    convert VOC mask file from .mat to .png, masked with id(1,2...20)
    :param work_path: base path of index file
    :param AUG_ori_index_path: file of 'path/to/img path/to/mask'
    :param AUG_converted_mask_dir: new mask files will locate here
    :return:
    """
    lines = open(AUG_ori_index_path, "r").readlines()
    for line in lines:
        mask_path = os.path.join(work_path, line.split()[1])
        mat = scipy.io.loadmat(mask_path, mat_dtype=True, squeeze_me=True, struct_as_record=False)
        mask = mat['GTcls'].Segmentation

        mask = Image.fromarray(mask, mode="L")
        mask_file_name = mask_path.split('/')[-1]
        mask_file_name = mask_file_name.split('.')[0]+'.png'
        out_path = os.path.join(AUG_converted_mask_dir, mask_file_name)
        mask.save(out_path)

def merge_index_and_image(work_path, img_merge_path, mask_merge_path,
                          AUG_ori_index_path, VOC_ori_index_path, out_path):
    """
    merge all data in augmented VOC and VOC 2012, and keep the data from the
    former when both have a same sample
    :param work_path: base path of the content of index file
    :param img_merge_path: the folder of merged data, containing images
    :param mask_merge_path: the folder of merged data, containing masks
    :param AUG_ori_index_path: 'path/to/img path/to/mask'
    :param VOC_ori_index_path: 'path/to/img path/to/mask'
    :param out_path: new index file
    :return:
    """
    merged_index = []
    lines = open(VOC_ori_index_path, "r").readlines()
    for line in lines:
        img_path = os.path.join(work_path, line.split()[0])
        img_name = img_path.split("/")[-1]
        merged_path = os.path.join(work_path, img_merge_path, img_name)
        shutil.copy(img_path, merged_path)
        merged_index.append(img_name.split('.')[0])
    lines = open(AUG_ori_index_path, "r").readlines()
    for line in lines:
        img_path = os.path.join(work_path, line.split()[0])
        img_name = img_path.split("/")[-1]
        merged_path = os.path.join(work_path, img_merge_path, img_name)
        shutil.copy(img_path, merged_path)
        merged_index.append(img_name.split('.')[0])
    merged_index = set(merged_index)
    out_file = open(out_path, "w+")
    for id in merged_index:
        img_name = id+".jpg"
        mask_name = id+".png"
        img_path = os.path.join(img_merge_path, img_name)
        mask_path = os.path.join(mask_merge_path, mask_name)
        out_file.write("%s %s\n" % (img_path, mask_path))
    out_file.close()


def decode_labels(mask, num_images=1, num_classes=21):
    """Decode batch of segmentation masks.

    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict (including background).

    Returns:
      A batch with num_images RGB images of the same size as the input.
    """
    n, h, w, c = mask.shape
    assert (n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (
    n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
        img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))  # Size is given as a (width, height)-tuple.
        pixels = img.load()
        for j_, j in enumerate(mask[i, :, :, 0]):
            for k_, k in enumerate(j):
                if k < num_classes:
                    pixels[k_, j_] = label_colours[k]
        outputs[i] = np.array(img)
    return outputs


def inv_preprocess(imgs, num_images, img_mean):
    """Inverse preprocessing of the batch of images.
       Add the mean vector and convert from BGR to RGB.

    Args:
      imgs: batch of input images.
      num_images: number of images to apply the inverse transformations on.
      img_mean: vector of mean colour values.

    Returns:
      The batch of the size num_images with the same spatial dimensions as the input.
    """
    n, h, w, c = imgs.shape
    assert (n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (
    n, num_images)
    outputs = np.zeros((num_images, h, w, c), dtype=np.uint8)
    for i in range(num_images):
        outputs[i] = (imgs[i] + img_mean)[:, :, ::-1].astype(np.uint8)
    return outputs


# convert_VOC_2012(work_path, VOC_ori_index_path, mask_out_dir)
# convert_VOC_Aug(work_path, AUG_ori_index_path, mask_out_dir)
converted_mask_dir = os.path.join("VOC_Seg", "mask")
converted_img_dir = os.path.join("VOC_Seg", "img")
merge_index_and_image(work_path, converted_img_dir, converted_mask_dir,
                      AUG_ori_index_path, VOC_ori_index_path, merged_index_path)
