import os
import random
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from skimage import color
from skimage.segmentation import find_boundaries

def r(strin, val):
    if val in strin:
        new_str = strin.replace(val, ' - ')
    else:
        new_str = strin
    return new_str


def show_image(img, title='', isGray=False, xticks=[], yticks=[], scale=1.0, isCv2=True):
    img_name = r(title, '\n')
    img_name = r(img_name, ':')
    if len(img.shape) == 3:
        if isCv2:
            img = img[:, :, ::-1]
    plt.figure(figsize=scale * plt.figaspect(1))
    plt.imshow(img, interpolation='nearest')
    if isGray:
        plt.gray()
    plt.title(title)
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.axis('off')
    plt.savefig(f'../output/{img_name}.jpg')
    plt.close()


def gen_SP(im, seg):
    range_Clust = np.unique(seg)
    dictt = {i: im[seg == i].mean(axis=0) / 255. for i in range_Clust}

    superpixels = np.zeros((seg.shape[0], seg.shape[1], 3))
    for i in range_Clust:
        superpixels[seg == i] = dictt[i]

    return superpixels


def Grad(im):
    grad = np.zeros((im.shape[0], im.shape[1]))

    for i in range(1, im.shape[0] - 1):
        for j in range(1, im.shape[1] - 1):
            x, y = [], []
            for i in range(3):
                x.append(im[i + 1, j, i] - im[i - 1, j, i])
                y.append(im[i, j + 1, i] - im[i, j - 1, i])

            grad[i, j] = np.linalg.norm(x) + np.linalg.norm(y)

    return grad


def relocate_cluster_center_at_lowgrad(grad, cluster_og):
    for i in range(len(cluster_og)):
        temp_arr = grad[cluster_og[i][0] - 3: cluster_og[i][0] + 3,
                   cluster_og[i][1] - 3: cluster_og[i][1] + 3]
        ind = np.where(temp_arr == temp_arr.min())
        updated = list(zip(ind[0], ind[1]))[0]
        cluster_og[i] = [cluster_og[i][0] - 3 + updated[0], cluster_og[i][1] - 3 + updated[1]]

    return cluster_og


def compute_res_error(m, S, lab, cluster, x, y):
    l1, a1, b1 = lab[x, y]
    x1, y1 = x, y
    l2, a2, b2 = lab[cluster[0], cluster[1]]
    x2, y2 = cluster[0], cluster[1]
    lab_distance = ((l1 - l2) ** 2 + (a1 - a2) ** 2 + (b1 - b2) ** 2) ** 0.5
    cood_distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    error = lab_distance + (m / S) * cood_distance
    return error


def SLIC(im, k, threshold, method, m):
    conv = {"regular": color.rgb2lab, "HSV": color.rgb2hsv, "CIE": color.rgb2rgbcie}
    N = im.shape[0] * im.shape[1]
    S = (N / k) ** 0.5
    lab = conv[method](im)
    cluster_og = []
    for i in range(int(S), lab.shape[0], int(S)):
        for j in range(int(S), lab.shape[1], int(S)):
            cluster_og.append([i, j])
    grad = Grad(lab)
    bool_val = True
    segs = np.zeros((im.shape[0], im.shape[1]))
    minClust = []
    cluster_og = relocate_cluster_center_at_lowgrad(grad, cluster_og)
    conv_error = 0.0
    init_error = 0.0
    for i in range(10000):
        if bool_val == True:
            for a in range(im.shape[0]):
                for b in range(im.shape[1]):
                    CloserClust = []
                    for c in cluster_og:
                        err = ((a - c[0]) ** 2 + (b - c[1]) ** 2) ** 0.5
                        if err <= 2 * S:
                            CloserClust.append(c)
                    if CloserClust:
                        err_arr = []
                        for c in CloserClust:
                            err = compute_res_error(m, S, lab, c, a, b)
                            err_arr.append(err)
                            if a == 0:
                                init_error = err
                            if err < threshold:
                                bool_val = False
                                if err != 0:
                                    conv_error = err
                                break

                        minClust = CloserClust[np.argmin(err_arr)]

                    for i in range(len(cluster_og)):
                        if list(cluster_og[i]) == list(minClust):
                            segs[a, b] = i

    return segs, init_error, conv_error


def getListofFiles(dir):
    inputs = []

    for path in os.listdir(dir):
        if os.path.isfile(os.path.join(dir, path)):
            inputs.append(path)
    return inputs


def getImages(dir):
    images = []
    inp = getListofFiles(dir)
    for i in range(len(inp)):
        temp = cv2.imread(dir + inp[i])
        images.append(temp)
    return images, inp


def calcTime(total_sec):
    hours = (int(total_sec / 3600))
    minutes = (int(((total_sec / 3600) - hours) * 60))
    seconds = (((((total_sec / 3600) - hours) * 60) - minutes) * 60)
    ret = ""
    if bool(hours):
        ret = ret + str("{:02d}".format(hours)) + "h "
    if bool(minutes):
        ret = ret + str("{:02d}".format(minutes)) + "m "
    if bool(seconds):
        ret = ret + str("{:0.2f}".format(seconds)) + "s"
    return ret


def getGT(dir, inp):
    gt = []
    for i in range(len(inp)):
        temp_name = inp[i].split('.')
        act_name = temp_name[0] + '.mat'
        inp[i] = act_name
    for i in range(len(inp)):
        temp1 = loadmat(dir + inp[i])
        temp = temp1['groundTruth']
        gt.append(temp)
    return gt


def getGTMap(gT):
    new = gT[0][0][0][0][1]
    for i in range(1, gT.shape[1]):
        new = new + gT[0][i][0][0][1]
    new = new / gT.shape[1]
    new[new > 0] = 255
    new[new <= 0] = 0
    return new


def calcBoundaryRecall(segs, gT):
    bound = find_boundaries(segs) * 255
    New = getGTMap(gT).astype(np.uint8)
    if bound.shape != New.shape:
        New = np.resize(New, bound.shape)
    fn = bound - New
    fn[fn >= 0] = 0
    fn = np.abs(fn)
    tp = bound + New

    tp = tp / 255
    fn = fn / 255
    tp[tp <= 1] = 0

    TP = np.sum(tp) / 2
    FN = np.sum(fn)
    bp = (TP) / (TP + FN)
    if bp >= 1:
        bp = bp / 10
    return round(bp * 100, 2)


def getbiggerSeg(gt):
    number_of_segs = []
    for i in range(gt.shape[1]):
        num = len(np.unique(gt[0][i][0][0][0]))
        number_of_segs.append(num)
    ind = number_of_segs.index(max(number_of_segs))
    biggerSeg = gt[0][ind][0][0][0]
    return biggerSeg


def genSeg(Seg, isGT):
    all_segs = []
    if isGT:
        seg = getbiggerSeg(Seg)
        for j in range(len(np.unique(seg))):
            temp = np.zeros(seg.shape).astype(np.byte)
            temp[np.where(seg == np.unique(seg)[j])] = 1
            all_segs.append(temp)
    else:
        for j in range(len(np.unique(Seg))):
            temp = np.zeros(Seg.shape).astype(np.byte)
            temp[np.where(Seg == np.unique(Seg)[j])] = 1
            all_segs.append(temp)
    return all_segs


def checkInter(seg1, seg2, in_, out_):
    if seg1.shape != seg2.shape:
        seg2 = np.resize(seg2, seg1.shape)
    seg_in = seg1 + seg2
    seg_in[seg_in < 2] = 0
    seg_in = seg_in / 2
    seg_out = np.abs(seg2 - seg_in)
    if np.sum(seg_in) != 0:
        in_.append(seg_in)
        out_.append(seg_out)
    return in_, out_


def getInt(gt_segs, pred_segs):
    seg_in1 = []
    seg_out1 = []
    for i in range(len(gt_segs)):
        for j in range(len(pred_segs)):
            seg_in1, seg_out1 = checkInter(gt_segs[i], pred_segs[j], seg_in1, seg_out1)
    return seg_in1, seg_out1


def calcUSerror(segs, gt):
    gt_segs = genSeg(gt, True)
    pred_segs = genSeg(segs, False)
    in_, out_ = getInt(gt_segs, pred_segs)
    N = segs.shape[0] * segs.shape[1]
    mini = []
    for i in range(len(in_)):
        in_val = np.sum(in_[i])
        out_val = np.sum(out_[i])
        if in_val < out_val:
            temp_min = np.sum(in_[i])
        else:
            temp_min = np.sum(out_[i])
        mini.append(temp_min.astype(np.byte))
    total_min = np.sum(np.asarray(mini)) * gt.shape[1]
    US_error = np.abs(total_min / N)
    if US_error >= 1:
        US_error = US_error / 10
    return round(US_error * 100, 2)


def randomImgs(img_list, val):
    ind = []
    for i in range(val):
        num = random.randint(0, len(img_list) - 1)
        ind.append(num)
    rnd_imgs = []
    for i in range(len(ind)):
        val = ind[i]
        rnd_imgs.append(img_list[val])
    return rnd_imgs


def differentWts(images):
    wts = [10, 50, 100]
    res = []
    method = 'regular'
    k = 64
    threshold = 0.5
    num_of_images = randomImgs(images, 1)
    og = num_of_images[0]
    for i in range(len(wts)):
        segs, _, _ = SLIC(og, k, threshold, method, wts[i])
        plot = gen_SP(og, segs)
        res.append(plot)

    return res, wts, k


def init_conv_error(images):
    wt = 10
    k = 64
    threshold = 0.5
    method = 'regular'
    num_imgs = randomImgs(images, 1)
    og = num_imgs[0]
    seg, init_error, conv_error = SLIC(og, k, threshold, method, wt)
    im1 = gen_SP(og, seg)
    print("Error at initialization: {:0.2f}".format(init_error))
    print("Error at convergence: {:0.2f}".format(conv_error))
    show_image(im1, f"SuperPixels with {k} clusters:\nInitialization error = {init_error}\nConvergence error = {conv_error}")
    return None


def calcRun(images):
    wt = 10
    k = [64, 256, 1024]
    thresh = 0.5
    method = 'regular'
    num_imgs = randomImgs(images, 1)
    og = num_imgs[0]
    for i in range(len(k)):
        start = time.perf_counter()
        seg, _, _ = SLIC(og, k[i], thresh, method, wt)
        stop = time.perf_counter()
        print(f"Time taken for {k[i]} clusters: {calcTime(stop - start)}")
        im_ = gen_SP(og, seg)
        show_image(im_, f"SuperPixels with {k[i]} clusters\nRuntime: {calcTime(stop-start)}")

    return None


def entireBSD(images, val):
    wt = 10
    k = [64, 256, 1024]
    thresh = 0.5
    method = 'regular'
    num_imgs = randomImgs(images, val)
    maps = []
    og_start = time.perf_counter()
    for i in range(len(num_imgs)):
        cluster_maps = []
        start = time.perf_counter()
        for j in range(len(k)):
            seg, _, _ = SLIC(num_imgs[i], k[j], thresh, method, wt)
            show_image(gen_SP(num_imgs[i], seg), f'Entire BSD - SuperPixels for image {i + 1} with {k[j]} clusters')
            cluster_maps.append(seg)
        stop = time.perf_counter()
        print(f'Time taken for image {i + 1}: {calcTime(stop - start)}')
        maps.append(cluster_maps)
    og_stop = time.perf_counter()
    print(f'Avg time taken for each image in the dataset: {calcTime((og_stop - og_start) / len(num_imgs))}')
    return maps


def errorCalc(maps, gt, title=''):
    for i in range(len(maps)):
        gT = gt[i]
        segMap = maps[i]

        for j in range(len(segMap)):
            bR = calcBoundaryRecall(segMap[j], gT)
            print(f"Boundary Recall Error for image {i + 1}{title}: {bR}%")
            us = calcUSerror(segMap[j], gT)
            print(f"Under-Segmentation Error for image {i + 1}{title}: {us}%\n")
    return None


def DiffMethod(images, val):
    wt = 10
    k = [256]
    thresh = 0.5
    methods = ["regular", "HSV", "CIE"]
    num_imgs = randomImgs(images, val)
    method_map = []
    maps = []
    for x in range(len(methods)):
        for i in range(len(num_imgs)):
            cluster_maps = []
            for j in range(len(k)):
                seg, _, _ = SLIC(num_imgs[i], k[j], thresh, methods[x], wt)
                show_image(gen_SP(num_imgs[i], seg),
                           f'SuperPixels for image {i + 1} with {k[j]} clusters using {methods[x]} method')
                cluster_maps.append(seg)
            maps.append(cluster_maps)
        method_map.append(maps)
    for a in range(len(method_map)):
        for b in range(len(method_map[a])):
            title = f' using {methods[a][b]} method'
            errorCalc(method_map[a][b], gt, title)
    return None


def varWts(images):
    imgs, diff_wts, k = differentWts(images)
    for i in range(len(imgs)):
        show_image(imgs[i], f"SuperPixels of {k} clusters with weight {diff_wts[i]}")
    return None


home = "../input/images/"
gt_dir = "../input/groundTruth/test/"
images, files = getImages(home)
gt = getGT(gt_dir, files)

choice = input("Please select the appropriate selection:\n"
               "1. Visualizing SLIC SuperPixels with varying weights\n"
               "2. Calculating initial and convergence error\n"
               "3. Calculate time for entire run\n"
               "4. Visualizing SLIC SuperPixels for the entire dataset\n"
               "5. Calculating the Boundary Recall and Undersegmentation error for the entire dataset\n"
               "6. Visualizing SLIC SuperPixels with other methods\n")

try:
    choice = int(choice)
except:
    print("Incorrect selection. Please select a number between 1 and 6")
    exit()

if choice == 1:
    varWts(images)
elif choice == 2:
    init_conv_error(images)
elif choice == 3:
    calcRun(images)
elif choice == 4:
    _ = entireBSD(images, 50)
elif choice == 5:
    maps = entireBSD(images, 50)
    errorCalc(maps, gt)
elif choice == 6:
    DiffMethod(images, 50)
else:
    print("Incorrect selection. Please select a number between 1 and 6")
    exit()
