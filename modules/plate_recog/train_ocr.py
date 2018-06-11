from multiprocessing.pool import ThreadPool
import os

import cv2 as cv

import numpy as np
from numpy.linalg import norm


# https://github.com/opencv/opencv/blob/master/samples/python/digits.py
# https://github.com/wzh191920/License-Plate-Recognition/blob/master/predict.py
SZ = 20 # size of each digit is SZ x SZ
CLASS_N = 10
DIGITS_FN = '../data/digits.png'

def deskew(img):
    m = cv.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv.warpAffine(img, M, (SZ, SZ), flags=cv.WARP_INVERSE_MAP | cv.INTER_LINEAR)
    return img


def preprocess_hog(digits):
    samples = []
    for img in digits:
        gx = cv.Sobel(img, cv.CV_32F, 1, 0)
        gy = cv.Sobel(img, cv.CV_32F, 0, 1)
        mag, ang = cv.cartToPolar(gx, gy)
        bin_n = 16
        bin = np.int32(bin_n*ang/(2*np.pi))
        bin_cells = bin[:10,:10], bin[10:,:10], bin[:10,10:], bin[10:,10:]
        mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
        hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)

        # transform to Hellinger kernel
        eps = 1e-7
        hist /= hist.sum() + eps
        hist = np.sqrt(hist)
        hist /= norm(hist) + eps

        samples.append(hist)
    return np.float32(samples)


class StatModel(object):
    def load(self, fn):
        self.model = self.model.load(fn)
    def save(self, fn):
        self.model.save(fn)


class SVM(StatModel):
    def __init__(self, kernel="rbf", C=1, degree=3, gamma=0.5):
        kernel_map = {
            "linear":cv.ml.SVM_LINEAR, "poly":cv.ml.SVM_POLY, "rbf":cv.ml.SVM_RBF
        }
        self.model = cv.ml.SVM_create()
        self.model.setKernel(kernel_map["rbf"])
        self.model.setType(cv.ml.SVM_C_SVC)
        self.model.setC(C)
        self.model.setDegree(degree)
        self.model.setGamma(gamma)

    def train(self, samples, labels):
        self.model.train(samples, cv.ml.ROW_SAMPLE, labels)

    def predict(self, samples):
        return self.model.predict(samples)[1].ravel()


class tune(object):
    def __init__(self):
        pass

    def cross_validate(self, model_class, params, samples, labels, kfold=3, pool=None):
        n = len(samples)
        shuffle_index = np.random.choice(range(n), n, replace=False)
        samples = samples[shuffle_index]
        labels = labels[shuffle_index]
        folds = np.array_split(np.arange(n), kfold)
        def f(i):
            model = model_class(**params)
            test_idx = folds[i]
            train_idx = list(folds)
            train_idx.pop(i)
            train_idx = np.hstack(train_idx)
            train_samples, train_labels = samples[train_idx], labels[train_idx]
            test_samples, test_labels = samples[test_idx], labels[test_idx]
            model.train(train_samples, train_labels)
            resp = model.predict(test_samples)
            score = (resp != test_labels).mean()
            print(".", end='')
            return score
        if pool is None:
            scores = list(map(f, range(kfold)))
        else:
            scores = pool.map(f, range(kfold))
        return np.mean(scores)

    def run_jobs(self, f, jobs):
        pool = ThreadPool(processes=int(cv.getNumberOfCPUs()*2/3))
        ires = pool.imap_unordered(f, jobs)
        return ires, pool

    def svm(self, samples, labels, kernel="rbf"):
        Cs = np.logspace(0, 10, 15, base=2)
        gammas = np.logspace(-7, 4, 15, base=2)
        scores = np.zeros((len(Cs), len(gammas)))
        scores[:] = np.nan

        print('adjusting SVM (may take a long time) ...')
        def f(job):
            i, j = job
            params = dict(kernel=kernel, C=Cs[i], gamma=gammas[j])
            score = self.cross_validate(SVM, params, samples, labels)
            print(i,j,score)
            return i, j, score

        ires, pool = self.run_jobs(f, np.ndindex(*scores.shape))
        for count, (i, j, score) in enumerate(ires):
            scores[i, j] = score
            print('%d / %d (best error: %.2f %%, last: %.2f %%)' %
                    (count+1, scores.size, np.nanmin(scores)*100, score*100))
        pool.close()
        print(scores)
#        print('writing score table to "svm_scores.npz"')
#        np.savez('svm_scores.npz', scores=scores, Cs=Cs, gammas=gammas)

        i, j = np.unravel_index(scores.argmin(), scores.shape)
        best_params = dict(C = Cs[i], gamma=gammas[j])
        print('best params:', best_params)
        print('best error: %.2f %%' % (scores.min()*100))

        self.best_model = SVM(kernel="rbf", **best_params)
        self.best_model.train(samples, labels)
        return best_params


if __name__ == '__main__':
    provinces = [
        "zh_cuan", "川", "zh_e", "鄂", "zh_gan", "赣", "zh_gan1", "甘", "zh_gui", "贵", 
        "zh_gui1", "桂", "zh_hei", "黑", "zh_hu", "沪", "zh_ji", "冀", "zh_jin", "津", 
        "zh_jing", "京", "zh_jl", "吉", "zh_liao", "辽", "zh_lu", "鲁", "zh_meng", "蒙", 
        "zh_min", "闽", "zh_ning", "宁", "zh_qing", "靑", "zh_qiong", "琼", "zh_shan", "陕",
        "zh_su", "苏", "zh_sx", "晋", "zh_wan", "皖", "zh_xiang", "湘", "zh_xin", "新",
        "zh_yu", "豫", "zh_yu1", "渝", "zh_yue", "粤", "zh_yun", "云", "zh_zang", "藏", "zh_zhe", "浙"
    ]

    x_chars, y_chars = [], []
    for root, dirs, files in os.walk("train/chars2"):
        root_base = os.path.basename(root)
        if root_base != "chars2":
            y_char = ord(root_base)
            x_chars += [cv.imread(os.path.join(root, f), cv.IMREAD_GRAYSCALE) for f in files]
            y_chars += [y_char] * len(files)
    x_chars = list(map(deskew, x_chars))
    x_chars = preprocess_hog(x_chars)
    y_chars = np.array(y_chars)
    adj = tune()
    adj.svm(x_chars, y_chars, kernel="rbf")
    adj.best_model.save("chars2.dat")

    x_chars, y_chars = [], []
    for root, dirs, files in os.walk("train/charsChinese"):
        root_base = os.path.basename(root)
        if root_base[:3] == "zh_":
            pinyin = os.path.basename(root)
            y_char = provinces.index(pinyin) + 1
            x_chars += [cv.imread(os.path.join(root, f), cv.IMREAD_GRAYSCALE) for f in files]
            y_chars += [y_char] * len(files)
    x_chars = list(map(deskew, x_chars))
    x_chars = preprocess_hog(x_chars)
    y_chars = np.array(y_chars)
    print(x_chars, y_chars)
    adj = tune()
    adj.svm(x_chars, y_chars, kernel="rbf")
    adj.best_model.save("charsChinese.dat")