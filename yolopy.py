import roslib
import rospy
from std_msgs.msg import Header
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge
import argparse
import numpy as np
import sys
import cv2
import os
import math
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException
from enum import Enum

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
class COCOLabels(Enum):
    vehicle = 0

class Param():
    def __init__(self, model='yolov5', width=640, height=640, url='localhost:8001', confidence=0.85, nms=0.45,
                 model_info=False, verbose=False, client_timeout=None):
        self.model = model
        self.width = width
        self.height = height
        self.url = url
        self.confidence = confidence
        self.nms = nms
        self.model_info = model_info
        self.verbose = verbose
        self.client_timeout = client_timeout

class BoundingBox:
    def __init__(self, classID, confidence, x1, x2, y1, y2, image_width, image_height):
        self.classID = classID
        self.confidence = confidence
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.u1 = x1 / image_width
        self.u2 = x2 / image_width
        self.v1 = y1 / image_height
        self.v2 = y2 / image_height
    def box(self):
        return (self.x1, self.y1, self.x2, self.y2)

    def width(self):
        return self.x2 - self.x1

    def height(self):
        return self.y2 - self.y1

    def center_absolute(self):
        return (0.5 * (self.x1 + self.x2), 0.5 * (self.y1 + self.y2))

    def center_normalized(self):
        return (0.5 * (self.u1 + self.u2), 0.5 * (self.v1 + self.v2))

    def size_absolute(self):
        return (self.x2 - self.x1, self.y2 - self.y1)

    def size_normalized(self):
        return (self.u2 - self.u1, self.v2 - self.v1)

def preprocess(raw_bgr_image, input_shape):
    input_w, input_h = input_shape
    image_raw = raw_bgr_image
    h, w, c = image_raw.shape
    image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
    # Calculate widht and height and paddings
    r_w = input_w / w
    r_h = input_h / h
    if r_h > r_w:
        tw = input_w
        th = int(r_w * h)
        tx1 = tx2 = 0
        ty1 = int((input_h - th) / 2)
        ty2 = input_h - th - ty1
    else:
        tw = int(r_h * w)
        th = input_h
        tx1 = int((input_w - tw) / 2)
        tx2 = input_w - tw - tx1
        ty1 = ty2 = 0
    # Resize the image with long side while maintaining ratio
    image = cv2.resize(image, (tw, th))
    # Pad the short side with (128,128,128)
    image = cv2.copyMakeBorder(
        image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (128, 128, 128)
    )
    image = image.astype(np.float32)
    # Normalize to [0,1]
    image /= 255.0
    # HWC to CHW format:
    image = np.transpose(image, [2, 0, 1])
    return image

def xywh2xyxy(x, origin_h, origin_w, input_w, input_h):
    y = np.zeros_like(x)
    r_w = input_w / origin_w
    r_h = input_h / origin_h
    if r_h > r_w:
        y[:, 0] = x[:, 0] - x[:, 2] / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2 - (input_h - r_w * origin_h) / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2 - (input_h - r_w * origin_h) / 2
        y /= r_w
    else:
        y[:, 0] = x[:, 0] - x[:, 2] / 2 - (input_w - r_h * origin_w) / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2 - (input_w - r_h * origin_w) / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2
        y /= r_h
    return y

def bbox_iou(box1, box2, x1y1x2y2=True):
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    # Get the coordinates of the intersection rectangle
    inter_rect_x1 = np.maximum(b1_x1, b2_x1)
    inter_rect_y1 = np.maximum(b1_y1, b2_y1)
    inter_rect_x2 = np.minimum(b1_x2, b2_x2)
    inter_rect_y2 = np.minimum(b1_y2, b2_y2)
    # Intersection area
    inter_area = np.clip(inter_rect_x2 - inter_rect_x1 + 1, 0, None) * \
                 np.clip(inter_rect_y2 - inter_rect_y1 + 1, 0, None)
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
    return iou

def non_max_suppression(prediction, origin_h, origin_w, input_w, input_h, conf_thres=0.5, nms_thres=0.1):
    # conf_thres = 0.01
    boxes = prediction[prediction[:, 4] >= conf_thres]
    # Trandform bbox from [center_x, center_y, w, h] to [x1, y1, x2, y2]
    boxes[:, :4] = xywh2xyxy(boxes[:, :4], origin_h, origin_w, input_w, input_h)
    # clip the coordinates
    boxes[:, 0] = np.clip(boxes[:, 0], 0, origin_w - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, origin_w - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, origin_h - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, origin_h - 1)
    # Object confidence
    confs = boxes[:, 4]
    # Sort by the confs
    boxes = boxes[np.argsort(-confs)]
    # Perform non-maximum suppression
    keep_boxes = []
    while boxes.shape[0]:
        large_overlap = bbox_iou(np.expand_dims(boxes[0, :4], 0), boxes[:, :4]) > nms_thres

        label_match = np.abs(boxes[0, -1] - boxes[:, -1]) < 2e-1
        # Indices of boxes with lower confidence scores, large IOUs and matching labels
        invalid = large_overlap & label_match
        keep_boxes += [boxes[0]]
        boxes = boxes[~invalid]
    boxes = np.stack(keep_boxes, 0) if len(keep_boxes) else np.array([])
    return boxes

def postprocess(output, origin_w, origin_h, input_shape, conf_th=0.5, nms_threshold=0.5, letter_box=False):
    output = output[0]
    boxes = non_max_suppression(output, origin_h, origin_w, input_shape[0], input_shape[1], conf_thres=conf_th,
                                nms_thres=nms_threshold)
    result_boxes = boxes[:, :4] if len(boxes) else np.array([])
    result_scores = boxes[:, 4] if len(boxes) else np.array([])
    result_classid = boxes[:, 5].astype(np.int) if len(boxes) else np.array([])
    detected_objects = []
    for box, score, label in zip(result_boxes, result_scores, result_classid):
        detected_objects.append(BoundingBox(label, score, box[0], box[2], box[1], box[3], origin_h, origin_w))
    return detected_objects

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    tl = (
            line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    if color == None:
        color = [np.random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))

    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )

def detect(img, cam_name):
    np.random.seed(69)
    RAND_COLORS = np.random.randint(10, 255, (80, 3), "int")
    inputs = []
    outputs = []
    print(f"before inference: {rospy.Time.now()}")
    inputs.append(grpcclient.InferInput('images', [1, 3, FLAGS.width, FLAGS.height], "FP32"))
    outputs.append(grpcclient.InferRequestedOutput('output0'))
    print(f"after inference: {rospy.Time.now()}")
    input_image = img  # cv2.imread(str(FLAGS.input))

    input_image_buffer = preprocess(input_image, [FLAGS.width, FLAGS.height])
    input_image_buffer = np.expand_dims(input_image_buffer, axis=0)
    inputs[0].set_data_from_numpy(input_image_buffer)

    # print("Invoking inference...")
    results = triton_client.infer(model_name=FLAGS.model,
                                  inputs=inputs,
                                  outputs=outputs,
                                  client_timeout=FLAGS.client_timeout)

    result = results.as_numpy('output0')

    detected_objects = postprocess(result, input_image.shape[1], input_image.shape[0], [FLAGS.width, FLAGS.height],
                                   FLAGS.confidence, FLAGS.nms)
    print(f"Detected objects: {len(detected_objects)}")
    # print("detection objects:",detected_objects[0])

    print(f"after postprocess: {rospy.Time.now()}")
    print("**********")

    if False:
        for box in detected_objects:
            print(f"{COCOLabels(box.classID).name}: {box.confidence}")
            plot_one_box(box.box(), input_image, color=tuple(RAND_COLORS[box.classID % 64].tolist()),
                         label=f"{COCOLabels(box.classID).name}:{box.confidence:.2f}", )
        cv2.imshow(cam_name, input_image)
        cv2.waitKey(3)
    return detected_objects

def image_callback_1(image, args):
    pub, cam_name = args[0], args[1]
    print(f"trigger callback: {rospy.Time.now()}")
    ros_image = np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.width, -1)
    ros_image = cv2.flip(ros_image, 0)
    ros_image = cv2.flip(ros_image, 1)

    detect_ret = detect(ros_image, cam_name)
    for box in detect_ret:
        ret_msg = TransformStamped()
        ret_msg.header.stamp = rospy.Time.now()
        ret_msg.transform.translation.x = box.classID
        ret_msg.transform.translation.y = box.confidence
        det_res = box.box()
        ret_msg.transform.rotation.x = det_res[0]
        ret_msg.transform.rotation.y = det_res[1]
        ret_msg.transform.rotation.z = det_res[2]
        ret_msg.transform.rotation.w = det_res[3]
        pub.publish(ret_msg)

if __name__ == '__main__':

    FLAGS = Param()
    print(FLAGS)
    # Create server context
    try:
        triton_client = grpcclient.InferenceServerClient(
            url=FLAGS.url,
            verbose=FLAGS.verbose,
            ssl=False,
            root_certificates=None,  # FLAGS.root_certificates,
            private_key=None,  # FLAGS.private_key,
            certificate_chain=None)  # FLAGS.certificate_chain)
    except Exception as e:
        print("context creation failed: " + str(e))
        sys.exit()

    # Health check
    if not triton_client.is_server_live():
        print("FAILED : is_server_live")
        sys.exit(1)

    if not triton_client.is_server_ready():
        print("FAILED : is_server_ready")
        sys.exit(1)

    if not triton_client.is_model_ready(FLAGS.model):
        print("FAILED : is_model_ready")
        sys.exit(1)

    if FLAGS.model_info:
        # Model metadata
        try:
            metadata = triton_client.get_model_metadata(FLAGS.model)
            print(metadata)
        except InferenceServerException as ex:
            if "Request for unknown model" not in ex.message():
                print("FAILED : get_model_metadata")
                print("Got: {}".format(ex.message()))
                sys.exit(1)
            else:
                print("FAILED : get_model_metadata")
                sys.exit(1)

        # Model configuration
        try:
            config = triton_client.get_model_config(FLAGS.model)
            if not (config.config.name == FLAGS.model):
                print("FAILED: get_model_config")
                sys.exit(1)
            print(config)
        except InferenceServerException as ex:
            print("FAILED : get_model_config")
            print("Got: {}".format(ex.message()))
            sys.exit(1)

    rospy.loginfo(os.system("pwd"))
    list_of_arguments = sys.argv
    '''
    模型初始化
    '''
    cam = list_of_arguments[1]
    rospy.init_node('ros_yolo_' + cam.split('/')[1])
    # rospy.get_param("/cam")
    image_topic_1 = cam  # "/cam1/image_raw"
    bbox_pub = rospy.Publisher('/detect_result_pub', TransformStamped, queue_size=10)
    rospy.Subscriber(image_topic_1, Image, image_callback_1, callback_args=(bbox_pub, cam,), queue_size=1,
                     buff_size=52428800)
    rospy.spin()