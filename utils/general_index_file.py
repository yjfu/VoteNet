import os
work_path = os.path.join("/", "home", "smartcar", "fuyj")
VOC_2012_seg_index_path = os.path.join(work_path, "VOCdevkit", "VOC2012",
                              "ImageSets", "Segmentation", "val.txt")
VOC_2012_image_base = os.path.join("VOCdevkit", "VOC2012",
                          "JPEGImages")
VOC_2012_mask_base = os.path.join("VOCdevkit", "VOC2012",
                            "SegmentationClass")
VOC_aug_seg_index_path = os.path.join(work_path, "benchmark_RELEASE",
                                      "dataset", "val.txt")
VOC_aug_image_base = os.path.join("benchmark_RELEASE", "dataset",
                                  "img")
VOC_aug_mask_base = os.path.join("benchmark_RELEASE", "dataset",
                                  "cls")

def generate_pair_index(seg_index_path, image_base,
                       mask_base, output_path, mask_ext="png"):
    lines = open(seg_index_path, "r").readlines()
    file = open(output_path, "w+")
    for line in lines:
        _line = line.split()[0]
        img_path = os.path.join(image_base, "%s.jpg" % _line)
        mask_path = os.path.join(mask_base, "%s.%s" % (_line, mask_ext))
        file.write("%s %s\n" % (img_path, mask_path))
    file.close()

def test_train(train_index_path, data_base):
    from matplotlib import pyplot as plt
    import matplotlib.image as mpimg
    index_file = open(train_index_path, "r")
    line = index_file.readline()
    img, mask = line.split()
    img = mpimg.imread(os.path.join(data_base, img))
    mask = mpimg.imread(os.path.join(data_base, mask))
    plt.imshow(img)
    plt.show()
    plt.imshow(mask)
    plt.show()

VOC_2012_output_path = os.path.join(work_path, "VoteNet", "data",
                           "voc_2012_valid.txt")
VOC_aug_output_path = os.path.join(work_path, "VoteNet", "data",
                           "voc_aug_valid.txt")
# generate VOC 2012 index
generate_pair_index(VOC_2012_seg_index_path, VOC_2012_image_base,
                    VOC_2012_mask_base, VOC_2012_output_path)
# generate VOC aug index
generate_pair_index(VOC_aug_seg_index_path, VOC_aug_image_base,
                    VOC_aug_mask_base, VOC_aug_output_path, "mat")

#test_train(train_output_path, work_path)


