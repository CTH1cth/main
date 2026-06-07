import numpy as np
from scipy.ndimage import convolve, distance_transform_edt as bwdist


_EPS = np.spacing(1)
_TYPE = np.float64


def _prepare_data(gt, pred):
    # COD 指标统一先把 GT 二值化、pred 归一化到 0..1。
    gt = np.asarray(gt)
    pred = np.asarray(pred)
    if gt.max() != gt.min():
        gt = (gt - gt.min()) / (gt.max() - gt.min())
    gt = gt > 0.5
    if pred.max() != pred.min():
        pred = (pred - pred.min()) / (pred.max() - pred.min())
    else:
        pred = pred.astype(int)
    return pred, gt


def _get_adaptive_threshold(matrix, max_value=1.0):
    # SOD/COD 常用 adaptive threshold：2 * mean，最大不超过 1。
    return min(2 * matrix.mean(), max_value)


class CODMetrics:
    def __init__(self):
        # 聚合 UCOD-DPL 使用的一组 COD 指标实现。
        self.mae = MAEmeasure()
        self.sm = Smeasure()
        self.em = Emeasure()
        self.fm = Fmeasure()
        self.wfm = WeightedFmeasure()
        self.acc = ACCmeasure()
        self.miou = IOUmeasure()

    def reset(self):
        self.mae.reset()
        self.sm.reset()
        self.em.reset()
        self.fm.reset()
        self.wfm.reset()
        self.acc.reset()
        self.miou.reset()

    def step(self, gt_tensor, pred_tensor):
        # 接收 torch tensor batch，逐样本转 numpy 后累计各指标。
        gt_tensor = gt_tensor.detach().cpu().numpy().astype(float)
        pred_tensor = pred_tensor.detach().cpu().numpy().astype(float)
        for i in range(gt_tensor.shape[0]):
            gt = gt_tensor[i].squeeze()
            pred = pred_tensor[i].squeeze()
            self.em.step(pred=pred, gt=gt)
            self.sm.step(pred=pred, gt=gt)
            self.fm.step(pred=pred, gt=gt)
            self.mae.step(pred=pred, gt=gt)
            self.wfm.step(pred=pred, gt=gt)
            self.acc.step(pred=pred, gt=gt)
            self.miou.step(pred=pred, gt=gt)

    def get_result(self):
        # 输出字段名与训练日志表格和 UCOD-DPL 指标名保持一致。
        em = self.em.get_results()["em"]
        fm = self.fm.get_results()["fm"]
        return {
            "ACC": self.acc.get_results()["acc"],
            "mIOU": self.miou.get_results()["miou"],
            "E_MAX": em["curve"].max(),
            "E_MEAN": em["curve"].mean(),
            "F_MAX": fm["curve"].max(),
            "F_MEAN": fm["curve"].mean(),
            "SMeasure": self.sm.get_results()["sm"],
            "MAE": self.mae.get_results()["mae"],
            "WFM": self.wfm.get_results()["wfm"],
        }


class ACCmeasure:
    def __init__(self):
        self.accs = []

    def reset(self):
        self.accs = []

    def step(self, pred, gt):
        # 二值 mask 的像素级 accuracy。
        pred, gt = _prepare_data(pred=pred, gt=gt)
        self.accs.append(float(np.sum(pred == gt) / gt.size))

    def get_results(self):
        return {"acc": np.mean(np.array(self.accs, _TYPE))}


class IOUmeasure:
    def __init__(self):
        self.ious = []

    def reset(self):
        self.ious = []

    def step(self, pred, gt):
        # 二值 mask 的 IoU，空 GT/空 pred 时按完全正确处理。
        pred, gt = _prepare_data(pred=pred, gt=gt)
        intersection = np.logical_and(pred, gt).sum()
        union = np.logical_or(pred, gt).sum()
        iou = 1.0 if union == 0 and intersection == 0 else intersection / union
        self.ious.append(float(iou))

    def get_results(self):
        return {"miou": np.mean(np.array(self.ious, _TYPE))}


class MAEmeasure:
    def __init__(self):
        self.maes = []

    def reset(self):
        self.maes = []

    def step(self, pred, gt):
        # MAE 使用归一化 pred 与二值 GT 的平均绝对误差。
        pred, gt = _prepare_data(pred=pred, gt=gt)
        mae = np.mean(np.abs(pred - gt))
        self.maes.append(mae)
        return 0, mae

    def get_results(self):
        return {"mae": np.mean(np.array(self.maes, _TYPE))}


class Smeasure:
    def __init__(self, alpha=0.5):
        self.sms = []
        self.alpha = alpha

    def reset(self):
        self.sms = []

    def step(self, pred, gt):
        pred, gt = _prepare_data(pred=pred, gt=gt)
        self.sms.append(self.cal_sm(pred, gt))

    def cal_sm(self, pred, gt):
        # S-measure 在全前景/全背景 GT 上使用定义中的特殊分支。
        y = np.mean(gt)
        if y == 0:
            sm = 1 - np.mean(pred)
        elif y == 1:
            sm = np.mean(pred)
        else:
            sm = self.alpha * self.object(pred, gt) + (1 - self.alpha) * self.region(pred, gt)
            sm = max(0, sm)
        return sm

    def object(self, pred, gt):
        # object-aware 项分别评价前景和背景区域。
        fg = pred * gt
        bg = (1 - pred) * (1 - gt)
        u = np.mean(gt)
        return u * self.s_object(fg, gt) + (1 - u) * self.s_object(bg, 1 - gt)

    def s_object(self, pred, gt):
        values = pred[gt == 1]
        if values.size <= 1:
            sigma_x = 0.0
        else:
            sigma_x = np.std(values, ddof=1)
        x = np.mean(values) if values.size else 0.0
        return 2 * x / (np.power(x, 2) + 1 + sigma_x + _EPS)

    def region(self, pred, gt):
        # region-aware 项按 GT 质心把图像切成四块计算结构相似度。
        x, y = self.centroid(gt)
        part_info = self.divide_with_xy(pred, gt, x, y)
        scores = [
            self.ssim(pred_part, gt_part)
            for pred_part, gt_part in zip(part_info["pred"], part_info["gt"])
        ]
        return sum(w * score for w, score in zip(part_info["weight"], scores))

    def centroid(self, matrix):
        # 空 GT 时使用图像中心，避免质心不可定义。
        h, w = matrix.shape
        if np.count_nonzero(matrix) == 0:
            x = np.round(w / 2)
            y = np.round(h / 2)
        else:
            y, x = np.argwhere(matrix).mean(axis=0).round()
        return int(x) + 1, int(y) + 1

    def divide_with_xy(self, pred, gt, x, y):
        h, w = gt.shape
        area = h * w
        gt_parts = (gt[0:y, 0:x], gt[0:y, x:w], gt[y:h, 0:x], gt[y:h, x:w])
        pred_parts = (pred[0:y, 0:x], pred[0:y, x:w], pred[y:h, 0:x], pred[y:h, x:w])
        weights = (
            x * y / area,
            y * (w - x) / area,
            (h - y) * x / area,
            1 - x * y / area - y * (w - x) / area - (h - y) * x / area,
        )
        return {"gt": gt_parts, "pred": pred_parts, "weight": weights}

    def ssim(self, pred, gt):
        h, w = pred.shape
        n = h * w
        if n <= 1:
            return 1.0
        x = np.mean(pred)
        y = np.mean(gt)
        sigma_x = np.sum((pred - x) ** 2) / (n - 1)
        sigma_y = np.sum((gt - y) ** 2) / (n - 1)
        sigma_xy = np.sum((pred - x) * (gt - y)) / (n - 1)
        alpha = 4 * x * y * sigma_xy
        beta = (x ** 2 + y ** 2) * (sigma_x + sigma_y)
        if alpha != 0:
            return alpha / (beta + _EPS)
        if beta == 0:
            return 1
        return 0

    def get_results(self):
        return {"sm": np.mean(np.array(self.sms, dtype=_TYPE))}


class Emeasure:
    def __init__(self):
        # 同时保存 adaptive E 和 256 阈值曲线 E。
        self.adaptive_ems = []
        self.changeable_ems = []

    def reset(self):
        self.adaptive_ems = []
        self.changeable_ems = []

    def step(self, pred, gt):
        pred, gt = _prepare_data(pred=pred, gt=gt)
        self.gt_fg_numel = np.count_nonzero(gt)
        self.gt_size = gt.shape[0] * gt.shape[1]
        changeable_ems = self.cal_changeable_em(pred, gt)
        adaptive_em = self.cal_adaptive_em(pred, gt)
        self.changeable_ems.append(changeable_ems)
        self.adaptive_ems.append(adaptive_em)
        return changeable_ems, adaptive_em

    def cal_adaptive_em(self, pred, gt):
        threshold = _get_adaptive_threshold(pred, max_value=1)
        return self.cal_em_with_threshold(pred, gt, threshold=threshold)

    def cal_changeable_em(self, pred, gt):
        return self.cal_em_with_cumsumhistogram(pred, gt)

    def cal_em_with_threshold(self, pred, gt, threshold):
        binarized_pred = pred >= threshold
        fg_fg_numel = np.count_nonzero(binarized_pred & gt)
        fg_bg_numel = np.count_nonzero(binarized_pred & ~gt)
        fg_numel = fg_fg_numel + fg_bg_numel
        bg_numel = self.gt_size - fg_numel

        if self.gt_fg_numel == 0:
            enhanced_matrix_sum = bg_numel
        elif self.gt_fg_numel == self.gt_size:
            enhanced_matrix_sum = fg_numel
        else:
            parts_numel, combinations = self.generate_parts_numel_combinations(
                fg_fg_numel=fg_fg_numel,
                fg_bg_numel=fg_bg_numel,
                pred_fg_numel=fg_numel,
                pred_bg_numel=bg_numel,
            )
            enhanced_matrix_sum = 0
            for part_numel, combination in zip(parts_numel, combinations):
                align_matrix_value = 2 * (combination[0] * combination[1]) / (
                    combination[0] ** 2 + combination[1] ** 2 + _EPS
                )
                enhanced_matrix_sum += ((align_matrix_value + 1) ** 2 / 4) * part_numel
        return enhanced_matrix_sum / (self.gt_size - 1 + _EPS)

    def cal_em_with_cumsumhistogram(self, pred, gt):
        pred = (pred * 255).astype(np.uint8)
        bins = np.linspace(0, 256, 257)
        fg_fg_hist, _ = np.histogram(pred[gt], bins=bins)
        fg_bg_hist, _ = np.histogram(pred[~gt], bins=bins)
        fg_fg_numel = np.cumsum(np.flip(fg_fg_hist), axis=0)
        fg_bg_numel = np.cumsum(np.flip(fg_bg_hist), axis=0)
        pred_fg_numel = fg_fg_numel + fg_bg_numel
        pred_bg_numel = self.gt_size - pred_fg_numel

        if self.gt_fg_numel == 0:
            enhanced_matrix_sum = pred_bg_numel
        elif self.gt_fg_numel == self.gt_size:
            enhanced_matrix_sum = pred_fg_numel
        else:
            parts_numel, combinations = self.generate_parts_numel_combinations(
                fg_fg_numel=fg_fg_numel,
                fg_bg_numel=fg_bg_numel,
                pred_fg_numel=pred_fg_numel,
                pred_bg_numel=pred_bg_numel,
            )
            results = np.empty(shape=(4, 256), dtype=np.float64)
            for i, (part_numel, combination) in enumerate(zip(parts_numel, combinations)):
                align_matrix_value = 2 * (combination[0] * combination[1]) / (
                    combination[0] ** 2 + combination[1] ** 2 + _EPS
                )
                results[i] = ((align_matrix_value + 1) ** 2 / 4) * part_numel
            enhanced_matrix_sum = results.sum(axis=0)
        return enhanced_matrix_sum / (self.gt_size - 1 + _EPS)

    def generate_parts_numel_combinations(self, fg_fg_numel, fg_bg_numel, pred_fg_numel, pred_bg_numel):
        bg_fg_numel = self.gt_fg_numel - fg_fg_numel
        bg_bg_numel = pred_bg_numel - bg_fg_numel
        parts_numel = [fg_fg_numel, fg_bg_numel, bg_fg_numel, bg_bg_numel]
        mean_pred_value = pred_fg_numel / self.gt_size
        mean_gt_value = self.gt_fg_numel / self.gt_size
        return parts_numel, [
            (1 - mean_pred_value, 1 - mean_gt_value),
            (1 - mean_pred_value, 0 - mean_gt_value),
            (0 - mean_pred_value, 1 - mean_gt_value),
            (0 - mean_pred_value, 0 - mean_gt_value),
        ]

    def get_results(self):
        adaptive_em = np.mean(np.array(self.adaptive_ems, dtype=_TYPE))
        changeable_em = np.mean(np.array(self.changeable_ems, dtype=_TYPE), axis=0)
        return {"em": {"adp": adaptive_em, "curve": changeable_em}}


class Fmeasure:
    def __init__(self, beta=0.3):
        self.beta = beta
        self.precisions = []
        self.recalls = []
        self.adaptive_fms = []
        self.changeable_fms = []

    def reset(self):
        self.precisions = []
        self.recalls = []
        self.adaptive_fms = []
        self.changeable_fms = []

    def step(self, pred, gt):
        pred, gt = _prepare_data(pred=pred, gt=gt)
        self.adaptive_fms.append(self.cal_adaptive_fm(pred=pred, gt=gt))
        precisions, recalls, changeable_fms = self.cal_pr(pred=pred, gt=gt)
        self.precisions.append(precisions)
        self.recalls.append(recalls)
        self.changeable_fms.append(changeable_fms)

    def cal_adaptive_fm(self, pred, gt):
        threshold = _get_adaptive_threshold(pred, max_value=1)
        binary_pred = pred >= threshold
        area_intersection = binary_pred[gt].sum()
        if area_intersection == 0:
            return 0
        pre = area_intersection / np.count_nonzero(binary_pred)
        rec = area_intersection / np.count_nonzero(gt)
        return (1 + self.beta) * pre * rec / (self.beta * pre + rec)

    def cal_pr(self, pred, gt):
        pred = (pred * 255).astype(np.uint8)
        bins = np.linspace(0, 256, 257)
        fg_hist, _ = np.histogram(pred[gt], bins=bins)
        bg_hist, _ = np.histogram(pred[~gt], bins=bins)
        fg_w_thrs = np.cumsum(np.flip(fg_hist), axis=0)
        bg_w_thrs = np.cumsum(np.flip(bg_hist), axis=0)
        tps = fg_w_thrs
        ps = fg_w_thrs + bg_w_thrs
        ps[ps == 0] = 1
        t = max(np.count_nonzero(gt), 1)
        precisions = tps / ps
        recalls = tps / t
        numerator = (1 + self.beta) * precisions * recalls
        denominator = np.where(numerator == 0, 1, self.beta * precisions + recalls)
        return precisions, recalls, numerator / denominator

    def get_results(self):
        adaptive_fm = np.mean(np.array(self.adaptive_fms, _TYPE))
        changeable_fm = np.mean(np.array(self.changeable_fms, dtype=_TYPE), axis=0)
        precision = np.mean(np.array(self.precisions, dtype=_TYPE), axis=0)
        recall = np.mean(np.array(self.recalls, dtype=_TYPE), axis=0)
        return {"fm": {"adp": adaptive_fm, "curve": changeable_fm}, "pr": {"p": precision, "r": recall}}


class WeightedFmeasure:
    def __init__(self, beta=1):
        self.beta = beta
        self.weighted_fms = []

    def reset(self):
        self.weighted_fms = []

    def step(self, pred, gt):
        pred, gt = _prepare_data(pred=pred, gt=gt)
        if np.all(~gt):
            wfm = 0
        else:
            wfm = self.cal_wfm(pred, gt)
        self.weighted_fms.append(wfm)

    def cal_wfm(self, pred, gt):
        dst, idx = bwdist(gt == 0, return_indices=True)
        error = np.abs(pred - gt)
        et = np.copy(error)
        et[gt == 0] = et[idx[0][gt == 0], idx[1][gt == 0]]
        kernel = self.matlab_style_gauss2d((7, 7), sigma=5)
        ea = convolve(et, weights=kernel, mode="constant", cval=0)
        min_error = np.where(gt & (ea < error), ea, error)
        b = np.where(gt == 0, 2 - np.exp(np.log(0.5) / 5 * dst), np.ones_like(gt))
        ew = min_error * b
        tpw = np.sum(gt) - np.sum(ew[gt == 1])
        fpw = np.sum(ew[gt == 0])
        r = 1 - np.mean(ew[gt == 1])
        p = tpw / (tpw + fpw + _EPS)
        return (1 + self.beta) * r * p / (r + self.beta * p + _EPS)

    def matlab_style_gauss2d(self, shape=(7, 7), sigma=5):
        m, n = [(ss - 1) / 2 for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h

    def get_results(self):
        return {"wfm": np.mean(np.array(self.weighted_fms, dtype=_TYPE))}
