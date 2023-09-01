import torch
import torch.nn as nn
import numpy as np
import math
import cv2

def box_ciou_angel_skewiou(b1, b2, box_loss_scale,sigma = 3):
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = torch.max(b1_mins, b2_mins)
    intersect_maxes = torch.min(b1_maxes, b2_maxes)
    intersect_wh = torch.max(intersect_maxes - intersect_mins, torch.zeros_like(intersect_maxes))
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    union_area = b1_area + b2_area - intersect_area
    iou = intersect_area / torch.clamp(union_area, min=1e-6)

    center_distance = torch.sum(torch.pow((b1_xy - b2_xy), 2), axis=-1)
    enclose_mins = torch.min(b1_mins, b2_mins)
    enclose_maxes = torch.max(b1_maxes, b2_maxes)
    enclose_wh = torch.max(enclose_maxes - enclose_mins, torch.zeros_like(intersect_maxes))
    enclose_diagonal = torch.sum(torch.pow(enclose_wh, 2), axis=-1)
    ciou = iou - 1.0 * (center_distance) / torch.clamp(enclose_diagonal, min=1e-6)

    v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(b1_wh[..., 0] / torch.clamp(b1_wh[..., 1], min=1e-6)) - torch.atan(
        b2_wh[..., 0] / torch.clamp(b2_wh[..., 1], min=1e-6))), 2)
    alpha = v / torch.clamp((1.0 - iou + v), min=1e-6)
    ciou = ciou - alpha * v
    ciou = (1 - ciou)*box_loss_scale

    sigma_squared = sigma * sigma
    reg_diff = torch.abs(b1[:, -1] - b2[:, -1])
    reg_loss = torch.where(
        torch.le(reg_diff, 1 / sigma_squared),
        0.5 * sigma_squared * torch.pow(reg_diff, 2),
        reg_diff - 0.5 / sigma_squared
    )

    total_loss = ciou + reg_loss

    return total_loss

def clip_by_tensor(t,t_min,t_max):
    t=t.float()
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result

def Focal_Loss(pred, target, gamma1= 1, gamma2= 1, alpha= 0.95):
    epsilon = 1e-7
    pred = clip_by_tensor(pred, epsilon, 1.0 - epsilon)
    output = -alpha*(1-pred)**gamma1*target * torch.log(pred) - \
             (1-alpha)*pred**gamma2*(1.0 - target) * torch.log(1.0 - pred)
    return output

def rotated_bbox_iou(gt_box, anchor_shapes):
    rect2 = ((gt_box[0][0], gt_box[0][1]), (gt_box[0][2], gt_box[0][3]), gt_box[0][4]*180/math.pi)
    anch_ious = torch.zeros(len(anchor_shapes))
    for i in range(len(anchor_shapes)):
        rect1 = ((anchor_shapes[i][0], anchor_shapes[i][1]),
                 (anchor_shapes[i][2], anchor_shapes[i][3]), anchor_shapes[i][4]*180/math.pi)
        r1 = cv2.rotatedRectangleIntersection(rect2, rect1)
        if r1[0] != 0:
            inter_area = cv2.contourArea(r1[1])
            union_area = rect1[1][0] * rect1[1][1] + rect2[1][0] * rect2[1][1] - inter_area + 1e-16
            iou = inter_area / union_area
            anch_ious[i] = iou
    return anch_ious

def AR_jaccard(_box_a, _box_b):
    b1_x1, b1_x2 = _box_a[:, 0] - _box_a[:, 2] / 2, _box_a[:, 0] + _box_a[:, 2] / 2
    b1_y1, b1_y2 = _box_a[:, 1] - _box_a[:, 3] / 2, _box_a[:, 1] + _box_a[:, 3] / 2
    b2_x1, b2_x2 = _box_b[:, 0] - _box_b[:, 2] / 2, _box_b[:, 0] + _box_b[:, 2] / 2
    b2_y1, b2_y2 = _box_b[:, 1] - _box_b[:, 3] / 2, _box_b[:, 1] + _box_b[:, 3] / 2
    box_a = torch.zeros_like(_box_a)
    box_b = torch.zeros_like(_box_b)
    box_a[:, 0], box_a[:, 1], box_a[:, 2], box_a[:, 3], box_a[:, 4] = b1_x1, b1_y1, b1_x2, b1_y2, _box_a[:, 4]
    box_b[:, 0], box_b[:, 1], box_b[:, 2], box_b[:, 3], box_b[:, 4] = b2_x1, b2_y1, b2_x2, b2_y2, _box_b[:, 4]
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:4].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:4].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    angle_offset = torch.abs(torch.cos(box_a[:, 4].unsqueeze(1).expand(A, B)
                                       -box_b[:, 4].unsqueeze(0).expand(A, B)))

    inter = inter[:, :, 0] * inter[:, :, 1]

    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)

    union = area_a + area_b - inter
    AR_iou = (inter / union)*angle_offset
    return AR_iou

class YOLOLoss(nn.Module):
    def __init__(self, anchors, angles, num_classes, img_size, label_smooth=0, cuda=True):
        super(YOLOLoss, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.angles = angles
        self.num_angles = len(angles)
        self.num_classes = num_classes

        self.bbox_attrs = 6
        self.img_size = img_size
        self.feature_length = [img_size[0 ]//32 ,img_size[0 ]//16 ,img_size[0 ]//8]
        self.label_smooth = label_smooth

        self.ignore_threshold = 0.5
        self.lambda_conf = 1.0
        self.lambda_cls = 1.0
        self.lambda_loc = 1.0
        self.cuda = cuda

    def forward(self, input, targets=None):
        bs = input.size(0)
        in_h = input.size(2)
        in_w = input.size(3)

        stride_h = self.img_size[1] / in_h
        stride_w = self.img_size[0] / in_w

        scaled_anchors = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors]
        angles = np.array(self.angles)
        prediction = input.view(bs, int(self.num_anchors/3), int(self.num_angles),
                                self.bbox_attrs, in_h, in_w).permute(0, 1, 2, 4, 5, 3).contiguous()


        conf = torch.sigmoid(prediction[..., 5])  
        mask, noobj_mask, t_box, tconf, box_loss_scale_x, box_loss_scale_y = self.get_target(targets,
                                                                                             scaled_anchors,
                                                                                             angles, in_w, in_h,
                                                                                             self.ignore_threshold)
        noobj_mask, pred_boxes_for_ciou = self.get_ignore(prediction, targets, scaled_anchors, angles, in_w, in_h, noobj_mask)

        if self.cuda:
            mask, noobj_mask = mask.cuda(), noobj_mask.cuda()
            box_loss_scale_x, box_loss_scale_y= box_loss_scale_x.cuda(), box_loss_scale_y.cuda()
            pred_boxes_for_ciou = pred_boxes_for_ciou.cuda()
            t_box = t_box.cuda()

        box_loss_scale = 2- box_loss_scale_x * box_loss_scale_y

        siou = box_ciou_angel_skewiou(pred_boxes_for_ciou[mask.bool()], t_box[mask.bool()],
                                                       box_loss_scale[mask.bool()])

        loss_loc = torch.sum(siou / bs)
        loss_conf = torch.sum(Focal_Loss(conf, mask) * mask / bs) + \
                    torch.sum(Focal_Loss(conf, mask) * noobj_mask / bs)

        loss = self.lambda_conf * loss_conf + loss_loc * self.lambda_loc

        return loss


    def get_target(self, target, anchors, angle, in_w, in_h, ignore_threshold):
        bs = len(target)
        anchor_index = [[0, 1, 2],
                        [3, 4, 5],
                        [6, 7, 8]][self.feature_length.index(in_w)]
        subtract_index = [0, 3, 6][self.feature_length.index(in_w)]

        mask = torch.zeros(bs, int(self.num_anchors / 3), int(self.num_angles), in_h, in_w, requires_grad=False)
        noobj_mask = torch.ones(bs, int(self.num_anchors / 3), int(self.num_angles), in_h, in_w, requires_grad=False)

        tx = torch.zeros(bs, int(self.num_anchors / 3), int(self.num_angles), in_h, in_w, requires_grad=False)
        ty = torch.zeros(bs, int(self.num_anchors / 3), int(self.num_angles), in_h, in_w, requires_grad=False)
        tw = torch.zeros(bs, int(self.num_anchors / 3), int(self.num_angles), in_h, in_w, requires_grad=False)
        th = torch.zeros(bs, int(self.num_anchors / 3), int(self.num_angles), in_h, in_w, requires_grad=False)

        tsita = torch.zeros(bs, int(self.num_anchors / 3), int(self.num_angles), in_h, in_w, requires_grad=False)

        t_box = torch.zeros(bs, int(self.num_anchors / 3), int(self.num_angles), in_h, in_w, 5, requires_grad=False)
        tconf = torch.zeros(bs, int(self.num_anchors / 3), int(self.num_angles), in_h, in_w, requires_grad=False)
        tcls = torch.zeros(bs, int(self.num_anchors / 3), int(self.num_angles), in_h, in_w, self.num_classes,
                           requires_grad=False)

        box_loss_scale_x = torch.zeros(bs, int(self.num_anchors / 3), int(self.num_angles), in_h, in_w,
                                       requires_grad=False)
        box_loss_scale_y = torch.zeros(bs, int(self.num_anchors / 3), int(self.num_angles), in_h, in_w,
                                       requires_grad=False)
        for b in range(bs):
            for t in range(target[b].shape[0]):
                gx = target[b][t, 0] * in_w
                gy = target[b][t, 1] * in_h

                gw = target[b][t, 2] * in_w
                gh = target[b][t, 3] * in_h

                gsita = target[b][t, 4]

                gi = int(gx)
                gj = int(gy)

                dlta = np.abs(gsita - angle)
                angle_n = np.argmin(dlta)
                select_angle = angle[angle_n]
                selected_angle = np.repeat(select_angle, self.num_anchors, axis=0)
                selected_angles = np.reshape(selected_angle, (-1, 1))

                gt_box = torch.FloatTensor(np.array([0, 0, gw, gh, gsita])).unsqueeze(0)
                anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((self.num_anchors, 2)),
                                                                  np.array(anchors), selected_angles), 1))

                anch_ious = rotated_bbox_iou(gt_box, anchor_shapes)
                best_n = np.argmax(anch_ious)
                if best_n not in anchor_index:
                    continue
                if (gj < in_h) and (gi < in_w):
                    best_n = best_n - subtract_index
                    noobj_mask[b, best_n, angle_n, gj, gi] = 0
                    mask[b, best_n, angle_n, gj, gi] = 1
                    tx[b, best_n, angle_n, gj, gi] = gx
                    ty[b, best_n, angle_n, gj, gi] = gy
                    tw[b, best_n, angle_n, gj, gi] = gw
                    th[b, best_n, angle_n, gj, gi] = gh
                    tsita[b, best_n, angle_n, gj, gi] = gsita

                    box_loss_scale_x[b, best_n, angle_n, gj, gi] = target[b][t, 2]
                    box_loss_scale_y[b, best_n, angle_n, gj, gi] = target[b][t, 3]

                    tconf[b, best_n, angle_n, gj, gi] = 1

                    tcls[b, best_n, angle_n, gj, gi, int(target[b][t, 5])] = 1
                else:
                    print('Step {0} out of bound'.format(b))
                    print('gj: {0}, height: {1} | gi: {2}, width: {3}'.format(gj, in_h, gi, in_w))
                    continue

        t_box[..., 0] = tx
        t_box[..., 1] = ty
        t_box[..., 2] = tw
        t_box[..., 3] = th
        t_box[..., 4] = tsita
        return mask, noobj_mask, t_box, tconf, box_loss_scale_x, box_loss_scale_y

    def get_ignore(self, prediction, target, scaled_anchors, angles, in_w, in_h, noobj_mask):
        bs = len(target)
        anchor_index = [[0, 1, 2], [3, 4, 5], [6, 7, 8]][self.feature_length.index(in_w)]
        scaled_anchors = np.array(scaled_anchors)[anchor_index]
        angles = np.array(angles).reshape(-1, 1)

        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        w = prediction[..., 2]  
        h = prediction[..., 3]  
        sita = prediction[..., 4]

        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

        grid_x = torch.linspace(0, in_w - 1, in_w).repeat(in_w, 1).repeat(
            int(bs * self.num_anchors / 3 * self.num_angles), 1, 1).view(x.shape).type(FloatTensor)
        grid_y = torch.linspace(0, in_h - 1, in_h).repeat(in_h, 1).t().repeat(
            int(bs * self.num_anchors / 3 * self.num_angles), 1, 1).view(y.shape).type(FloatTensor)


        anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
        anchor_angle = FloatTensor(angles).index_select(1, LongTensor([0]))

        anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w * self.num_angles).view(w.shape)
        anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w * self.num_angles).view(h.shape)

        anchor_angle = anchor_angle.repeat(bs * int(self.num_anchors / 3), 1).repeat(1, 1, in_h * in_w).view(sita.shape)
        pred_boxes = FloatTensor(prediction[..., :5].shape)
        pred_boxes[..., 0] = x + grid_x
        pred_boxes[..., 1] = y + grid_y
        pred_boxes[..., 2] = torch.exp(w) * anchor_w
        pred_boxes[..., 3] = torch.exp(h) * anchor_h
        pred_boxes[..., 4] = sita + anchor_angle

        for i in range(bs):
            pred_boxes_for_ignore = pred_boxes[i]
            pred_boxes_for_ignore = pred_boxes_for_ignore.view(-1, 5)
            if len(target[i]) > 0:
                gx = target[i][:, 0:1] * in_w
                gy = target[i][:, 1:2] * in_h
                gw = target[i][:, 2:3] * in_w
                gh = target[i][:, 3:4] * in_h
                gsita = target[i][:, 4:5]

                gt_box = torch.FloatTensor(np.concatenate([gx, gy, gw, gh, gsita], -1)).type(FloatTensor)
                anch_ious = AR_jaccard(gt_box, pred_boxes_for_ignore)

                for t in range(target[i].shape[0]):
                    anch_iou = anch_ious[t].view(pred_boxes[i].size()[:4])
                    noobj_mask[i][anch_iou > 0.3] = 0

        return noobj_mask, pred_boxes
