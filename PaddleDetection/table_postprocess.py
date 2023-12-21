import numpy as np 
from scipy.special import softmax
import cv2 
import math
from table_preprocess import TablePreprocess



def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    """
    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
        candidate_size: only consider the candidates with the highest scores.
    Returns:
         picked: a list of indexes of the kept boxes
    """
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    indexes = np.argsort(scores)
    indexes = indexes[-candidate_size:]
    while len(indexes) > 0:
        current = indexes[-1]
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[:-1]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            np.expand_dims(
                current_box, axis=0), )
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]


def iou_of(boxes0, boxes1, eps=1e-5):
    """Return intersection-over-union (Jaccard index) of boxes.
    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)


def area_of(left_top, right_bottom):
    """Compute the areas of rectangles given two corners.
    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.
    Returns:
        area (N): return the area.
    """
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]


class PicoDetPostProcess(object):
    """
    Args:
        input_shape (int): network input image size
        ori_shape (int): ori image shape of before padding
        scale_factor (float): scale factor of ori image
        enable_mkldnn (bool): whether to open MKLDNN
    """

    def __init__(self,
                 input_shape,
                 ori_shape,
                 scale_factor,
                 strides=[8, 16, 32, 64],
                 score_threshold=0.4,
                 nms_threshold=0.5,
                 nms_top_k=1000,
                 keep_top_k=100):
        self.ori_shape = ori_shape
        self.input_shape = input_shape
        self.scale_factor = scale_factor
        self.strides = strides
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.nms_top_k = nms_top_k
        self.keep_top_k = keep_top_k

    def warp_boxes(self, boxes, ori_shape):
        """Apply transform to boxes
        """
        width, height = ori_shape[1], ori_shape[0]
        n = len(boxes)
        if n:
            # warp points
            xy = np.ones((n * 4, 3))
            xy[:, :2] = boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
                n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            # xy = xy @ M.T  # transform
            xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate(
                (x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
            # clip boxes
            xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
            xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
            return xy.astype(np.float32)
        else:
            return boxes

    def __call__(self, scores, raw_boxes):
        batch_size = raw_boxes[0].shape[0]
        reg_max = int(raw_boxes[0].shape[-1] / 4 - 1)
        out_boxes_num = []
        out_boxes_list = []
        for batch_id in range(batch_size):
            # generate centers
            decode_boxes = []
            select_scores = []
            for stride, box_distribute, score in zip(self.strides, raw_boxes,
                                                     scores):
                box_distribute = box_distribute[batch_id]
                score = score[batch_id]
                # centers
                fm_h = self.input_shape[0] / stride
                fm_w = self.input_shape[1] / stride
                h_range = np.arange(fm_h)
                w_range = np.arange(fm_w)
                ww, hh = np.meshgrid(w_range, h_range)
                ct_row = (hh.flatten() + 0.5) * stride
                ct_col = (ww.flatten() + 0.5) * stride
                center = np.stack((ct_col, ct_row, ct_col, ct_row), axis=1)

                # box distribution to distance
                reg_range = np.arange(reg_max + 1)
                box_distance = box_distribute.reshape((-1, reg_max + 1))
                box_distance = softmax(box_distance, axis=1)
                box_distance = box_distance * np.expand_dims(reg_range, axis=0)
                box_distance = np.sum(box_distance, axis=1).reshape((-1, 4))
                box_distance = box_distance * stride

                # top K candidate
                topk_idx = np.argsort(score.max(axis=1))[::-1]
                topk_idx = topk_idx[:self.nms_top_k]
                center = center[topk_idx]
                score = score[topk_idx]
                box_distance = box_distance[topk_idx]

                # decode box
                decode_box = center + [-1, -1, 1, 1] * box_distance

                select_scores.append(score)
                decode_boxes.append(decode_box)

            # nms
            bboxes = np.concatenate(decode_boxes, axis=0)
            confidences = np.concatenate(select_scores, axis=0)
            picked_box_probs = []
            picked_labels = []
            for class_index in range(0, confidences.shape[1]):
                probs = confidences[:, class_index]
                mask = probs > self.score_threshold
                probs = probs[mask]
                if probs.shape[0] == 0:
                    continue
                subset_boxes = bboxes[mask, :]
                box_probs = np.concatenate(
                    [subset_boxes, probs.reshape(-1, 1)], axis=1)
                box_probs = hard_nms(
                    box_probs,
                    iou_threshold=self.nms_threshold,
                    top_k=self.keep_top_k, )
                picked_box_probs.append(box_probs)
                picked_labels.extend([class_index] * box_probs.shape[0])

            if len(picked_box_probs) == 0:
                out_boxes_list.append(np.empty((0, 4)))
                out_boxes_num.append(0)

            else:
                picked_box_probs = np.concatenate(picked_box_probs)

                # resize output boxes
                picked_box_probs[:, :4] = self.warp_boxes(
                    picked_box_probs[:, :4], self.ori_shape[batch_id])
                im_scale = np.concatenate([
                    self.scale_factor[batch_id][::-1],
                    self.scale_factor[batch_id][::-1]
                ])
                picked_box_probs[:, :4] /= im_scale
                # clas score box
                out_boxes_list.append(
                    np.concatenate(
                        [
                            np.expand_dims(
                                np.array(picked_labels),
                                axis=-1), np.expand_dims(
                                    picked_box_probs[:, 4], axis=-1),
                            picked_box_probs[:, :4]
                        ],
                        axis=1))
                out_boxes_num.append(len(picked_labels))

        out_boxes_list = np.concatenate(out_boxes_list, axis=0)
        out_boxes_num = np.asarray(out_boxes_num).astype(np.int32)
        return out_boxes_list, out_boxes_num


class Detector(object):
    def __init__(self,predictor,
                 nms={"keep_top_k": 100, "name": "MultiClassNMS", "nms_threshold": 0.5, "nms_top_k": 1000, "score_threshold": 0.4},
                 fpn_stride=[8, 16, 32, 64],
                 mask=False):
        self.pred_config = {"nms":nms, "fpn_stride":fpn_stride ,"mask":mask}
        self.predictor = predictor
        self.table_preprocess = TablePreprocess()


    def preprocess(self, image_list):
        inputs = self.table_preprocess(image_list)
        return inputs

    def postprocess(self, inputs, result):
        # postprocess output of predictor
        np_boxes_num = result["boxes_num"]
        assert isinstance(np_boxes_num, np.ndarray), \
            "`np_boxes_num` should be a `numpy.ndarray`"

        result = {k: v for k, v in result.items() if v is not None}
        return result

    def predict(self, repeats=1):
        """
        Args:
            repeats (int): repeats number for prediction
        Returns:
            result (dict): include "boxes": np.ndarray: shape:[N,6], N: number of box,
                            matix element:[class, score, x_min, y_min, x_max, y_max]
                            MaskRCNN"s result include "masks": np.ndarray:
                            shape: [N, im_h, im_w]
        """
        # model prediction
        np_boxes_num, np_boxes, np_masks = np.array([0]), None, None


        for i in range(repeats):
            self.predictor.run()
            output_names = self.predictor.get_output_names()
            boxes_tensor = self.predictor.get_output_handle(output_names[0])
            np_boxes = boxes_tensor.copy_to_cpu()
            if len(output_names) == 1:
                # some exported model can not get tensor "bbox_num" 
                np_boxes_num = np.array([len(np_boxes)])
            else:
                boxes_num = self.predictor.get_output_handle(output_names[1])
                np_boxes_num = boxes_num.copy_to_cpu()
            if self.pred_config.mask:
                masks_tensor = self.predictor.get_output_handle(output_names[2])
                np_masks = masks_tensor.copy_to_cpu()
        result = dict(boxes=np_boxes, masks=np_masks, boxes_num=np_boxes_num)
        return result

    def merge_batch_result(self, batch_result):
        if len(batch_result) == 1:
            return batch_result[0]
        res_key = batch_result[0].keys()
        results = {k: [] for k in res_key}
        for res in batch_result:
            for k, v in res.items():
                results[k].append(v)
        for k, v in results.items():
            if k not in ["masks", "segm"]:
                results[k] = np.concatenate(v)
        return results

    def predict_image(self, image_list):
        batch_loop_cnt = math.ceil(float(len(image_list)) / self.batch_size)
        results = []
        for i in range(batch_loop_cnt):
            start_index = i * self.batch_size
            end_index = min((i + 1) * self.batch_size, len(image_list))
            batch_image_list = image_list[start_index:end_index]
        
            # preprocess
            inputs = self.preprocess(batch_image_list)

            # model prediction
            result = self.predict()

            # postprocess
            result = self.postprocess(inputs, result)
                
            results.append(result)
            print("Test iter {}".format(i))
        results = self.merge_batch_result(results)
        return results


class DetectorPicoDet(Detector):
    def __init__(self, predictor):
        super(DetectorPicoDet, self).__init__(predictor=predictor)
        self.input_name, self.output_names = self.get_in_output_names()
        
    def get_in_output_names(self):
        
        outputs = self.predictor.get_outputs()
        output_name = [x.name for x in outputs]
        input_name = self.predictor.get_inputs()[0].name

        return input_name, output_name
    
    def postprocess(self, inputs, result):
        # postprocess output of predictor
        np_score_list = result["boxes"]
        np_boxes_list = result["boxes_num"]
        postprocessor = PicoDetPostProcess(
            inputs["image"].shape[2:],
            inputs["im_shape"],
            inputs["scale_factor"],
            strides=self.pred_config['fpn_stride'],
            nms_threshold=self.pred_config["nms"]["nms_threshold"],
            score_threshold = self.pred_config["nms"]["score_threshold"])
        np_boxes, np_boxes_num = postprocessor(np_score_list, np_boxes_list)
        result = dict(boxes=np_boxes, boxes_num=np_boxes_num)
        return result

    def predict(self, inputs):
        """
        Args:
            repeats (int): repeat number for prediction
        Returns:
            result (dict): include "boxes": np.ndarray: shape:[N,6], N: number of box,
                            matix element:[class, score, x_min, y_min, x_max, y_max]
        """
        np_score_list, np_boxes_list = [], []

        outputs = self.predictor.run(self.output_names,{self.input_name:inputs})
        
        num_outs = int(len(self.output_names) / 2)
        for out_idx in range(num_outs):
            np_score_list.append(outputs[out_idx])
            np_boxes_list.append(outputs[out_idx + num_outs])
        result = dict(boxes=np_score_list, boxes_num=np_boxes_list)
        return result
    
    def __call__(self, image):
        inputs = self.preprocess(image)
        result = self.predict(inputs[self.input_name])
        result = self.postprocess(inputs, result)
        return result


def onnx_infer_test():
    onnx_path = "table_detect_model.onnx"
    import onnxruntime as ort
    ort_sess = ort.InferenceSession(onnx_path)
    input_name = ort_sess.get_inputs()
    outputs = ort_sess.get_outputs()
    
    output_name = [x.name for x in outputs]
    
    table_pre = TablePreprocess()
    img = cv2.imread("docs/images/layout.jpg")
    inputs  = table_pre(img)
    print(inputs["image"].shape)
    

    print(output_name)
    for i in range(1):
        output_infer = ort_sess.run(output_name, {input_name[0].name: inputs["image"]})
        print([x.shape for x in output_infer])
        for o in output_infer:
            print(o)
            print("*"*30)
if __name__ == "__main__":
    onnx_path = "table_detect_model.onnx"
    import onnxruntime as ort
    ort_sess = ort.InferenceSession(onnx_path)
    table_detector = DetectorPicoDet(predictor=ort_sess)
    
    img = cv2.imread("6.jpg")
    print(img.shape)
    result = table_detector(img)
    print(result)
    # onnx_infer_test()