import os
import dlib
import cv2
import copy
import numpy as np

from PIL import Image, ImageDraw
from .constants import script_models_dir
from .utils import expand_polygon, calculate_distance
from .modules import depthmap, mmdetdd

depthmap_generator = depthmap.SimpleDepthMapGenerator()

try:
    landmark_detector = dlib.shape_predictor(
        os.path.join(script_models_dir, 'shape_predictor_68_face_landmarks.dat')
    )
except Exception as e:
    # when reloading the module, landmark_detector is already loaded
    # and the file cant be opened again
    pass

def _get_image_mask(image):
    width, height = image.size
    return np.full((height, width, 3), 0, dtype=np.uint8)

def _get_detected_faces_dlib(image):

    # Load the pre-trained model for detecting facial landmarks
    face_detector = dlib.get_frontal_face_detector()

    # Convert the image to grayscale
    frame = np.array(image)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    return face_detector(gray, 1), gray

def _calculate_mask_padding(p1, p2, percents):
    distance = calculate_distance((p1.x, p1.y), (p2.x, p2.y))
    value = int(distance * (percents / 100))
    return value

def get_eyes_mask_dlib(init_image, padding=20, in_pixels=False):

    mask = _get_image_mask(init_image)
    faces, gray = _get_detected_faces_dlib(init_image)
    padding_original = padding

    if not len(faces):
        return Image.fromarray(mask), False

    for face in faces:
        # Use the landmark detector to find facial landmarks
        landmarks = landmark_detector(gray, face)
        # Extract the coordinates of the left and right eyes
        left_eye_x = [landmarks.part(i).x for i in range(36, 42)]
        left_eye_y = [landmarks.part(i).y for i in range(36, 42)]
        right_eye_x = [landmarks.part(i).x for i in range(42, 48)]
        right_eye_y = [landmarks.part(i).y for i in range(42, 48)]
        # Calculate mask padding
        if not in_pixels:
            padding = _calculate_mask_padding(landmarks.part(36), landmarks.part(39), padding_original)
        # Draw a filled white polygon around the left eye
        left_eye_points = []
        for i in range(len(left_eye_x)):
            left_eye_points.append([left_eye_x[i], left_eye_y[i]])
        left_eye_points = expand_polygon(left_eye_points, padding)
        left_eye_points = np.array(left_eye_points, np.int32)
        left_eye_points = left_eye_points.reshape((-1, 1, 2))
        cv2.fillPoly(mask, [left_eye_points], (255, 255, 255))
        # Calculate mask padding
        if not in_pixels:
            padding = _calculate_mask_padding(landmarks.part(42), landmarks.part(45), padding_original)
        # Draw a filled white polygon around the right eye
        right_eye_points = []
        for i in range(len(right_eye_x)):
            right_eye_points.append([right_eye_x[i], right_eye_y[i]])
        right_eye_points = expand_polygon(right_eye_points, padding)
        right_eye_points = np.array(right_eye_points, np.int32)
        right_eye_points = right_eye_points.reshape((-1, 1, 2))
        cv2.fillPoly(mask, [right_eye_points], (255, 255, 255))

    return Image.fromarray(mask), True

def get_face_mask_dlib(init_image, padding=20, in_pixels=False):

    mask = _get_image_mask(init_image)
    faces, gray = _get_detected_faces_dlib(init_image)
    padding_original = padding

    if not len(faces):
        return Image.fromarray(mask), False

    for face in faces:
        # Use the landmark detector to find facial landmarks
        landmarks = landmark_detector(gray, face)
        face_x = [landmarks.part(i).x for i in range(0, 17)] + [landmarks.part(i).x for i in reversed(range(17, 27))]
        face_y = [landmarks.part(i).y for i in range(0, 17)] + [landmarks.part(i).y for i in reversed(range(17, 27))]
        # Calculate mask padding
        if not in_pixels:
            padding = _calculate_mask_padding(landmarks.part(0), landmarks.part(16), padding_original)
        # Draw a filled white polygon around the face
        face_points = []
        for i in range(len(face_x)):
            face_points.append([face_x[i], face_y[i]])
        face_points = expand_polygon(face_points, padding)
        face_points = np.array(face_points, np.int32)
        face_points = face_points.reshape((-1, 1, 2))
        cv2.fillPoly(mask, [face_points], (255, 255, 255))

    return Image.fromarray(mask), True

def get_face_mask_depth(init_image):

    mask = _get_image_mask(init_image)
    faces, gray = _get_detected_faces_dlib(init_image)

    if len(faces) != 1:
        return Image.fromarray(mask), False

    body_mask, body_mask_success = get_body_mask_depth(init_image)

    if not body_mask_success:
        return Image.fromarray(mask), False

    face = faces[0]
    landmarks = landmark_detector(gray, face)
    lowest_point = None

    for i in range(0, 17):
        point = landmarks.part(i)
        if lowest_point is None or point.y > lowest_point.y:
            lowest_point = point

    width, height = body_mask.size

    for x in range(0, width):
        for y in range(lowest_point.y, height):
            body_mask.putpixel((x, y), (0, 0, 0))

    return body_mask, True

def get_body_mask_depth(init_image):

    def remap_range(value, minIn, MaxIn, minOut, maxOut):
        if value > MaxIn: value = MaxIn;
        if value < minIn: value = minIn;
        finalValue = ((value - minIn) / (MaxIn - minIn)) * (maxOut - minOut) + minOut
        return finalValue

    def create_depth_mask_from_depth_map(img, treshold, clean_cut):
        img = copy.deepcopy(img.convert("RGBA"))
        mask_img = copy.deepcopy(img.convert("L"))
        mask_datas = mask_img.getdata()
        datas = img.getdata()
        newData = []
        maxD = max(mask_datas)
        if clean_cut and treshold == 0:
            treshold = 128
        for i in range(len(mask_datas)):
            if clean_cut and mask_datas[i] > treshold:
                newrgb = 255
            elif mask_datas[i] > treshold and not clean_cut:
                newrgb = int(remap_range(mask_datas[i],treshold,255,0,255))
            else:
                newrgb = 0
            newData.append((newrgb,newrgb,newrgb,255))
        img.putdata(newData)
        return img

    try:
        d_m = depthmap_generator.calculate_depth_maps(init_image, init_image.width, init_image.height, 1, False)
        d_m = create_depth_mask_from_depth_map(d_m, 128, True)
        return d_m, True
    except Exception as e:
        print(e)
        return _get_image_mask(init_image), False

def get_face_mask_mmdet(init_image):

    mask = _get_image_mask(init_image)
    faces, gray = _get_detected_faces_dlib(init_image)

    if len(faces) != 1:
        return Image.fromarray(mask), False

    body_mask, body_mask_success = get_body_mask_mmdet(init_image)

    if not body_mask_success:
        return Image.fromarray(mask), False

    face = faces[0]
    landmarks = landmark_detector(gray, face)
    lowest_point = None

    for i in range(0, 17):
        point = landmarks.part(i)
        if lowest_point is None or point.y > lowest_point.y:
            lowest_point = point

    width, height = body_mask.size

    for x in range(0, width):
        for y in range(lowest_point.y, height):
            body_mask.putpixel((x, y), 0)

    return body_mask, True

def get_body_mask_mmdet(init_image):
    try:
        mask = mmdetdd.get_person_mask(init_image)
        return mask, True
    except Exception as e:
        print(e)
    return _get_image_mask(init_image), False
