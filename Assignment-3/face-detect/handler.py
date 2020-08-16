try:
    import unzip_requirements
except ImportError:
    pass
from requests_toolbelt.multipart import decoder
from PIL import Image
import boto3
import os
import io
import base64
import json
import dlib
import numpy as np
import math
import faceBlendCommon as fbc
import faceswap as fsp
import boto3
# from renderFace import renderFace
# import matplotlib.pyplot as plt
# import faceBlendCommon as fbc
# from dataPath import DATA_PATH
# from dataPath import MODEL_PATH
# landmark model location
PREDICTOR_PATH_5 =  "predictor/shape_predictor_5_face_landmarks.dat"
PREDICTOR_PATH_68 = "predictor/shape_predictor_68_face_landmarks.dat"
# S3_BUCKET = os.environ['S3_BUCKET'] if 'S3_BUCKET' in os.environ else 'mobilenetv2evas1'
# MODEL_PATH = os.environ['MODEL_PATH'] if 'MODEL_PATH' in os.environ else 'shape_predictor_68_face_landmarks.dat'


# s3 = boto3.client('s3')


# try:
#     if os.path.isfile(MODEL_PATH) !=True:
#         s3.download_file('mobilenetv2evas1','shape_predictor_68_face_landmarks.dat','shape_predictor_68_face_landmarks.dat')
# except Exception as e:
#     print(repr(e))
#     raise(e)


faceDetector = dlib.get_frontal_face_detector()

landmarkDetector5 = dlib.shape_predictor(PREDICTOR_PATH_5)

def transform_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes))
        return np.array(image)
    except Exception as e:
        print(repr(e))
        raise(e)

# imageFilename = DATA_PATH + "images/family.jpg"
# im = cv2.imread(imageFilename)
# landmarksBasename = "results/family"
# plt.imshow(im[:,:,::-1])
# plt.show()

def align_face(bytes_im):
    im= transform_image(image_bytes=bytes_im)

    points = fbc.getLandmarks(faceDetector, landmarkDetector5, im)

    points = np.array(points)

    im = np.float32(im)/255.0

    # Dimensions of the output image
    h = im.shape[0]
    w = im.shape[1]

    imNorm, points = fbc.normalizeImagesAndLandmarks((h,w), im, points)

    imNorm = np.uint8(imNorm*255)

    return imNorm


def img_to_base64(img):
    img = Image.fromarray(img, 'RGB') 
    buffer = io.BytesIO()
    img.save(buffer,format="JPEG")
    decodedimage = buffer.getvalue()                     
    img_64 = f"data:image/jpeg;base64,{base64.b64encode(decodedimage).decode()}"
    return img_64


# Detect faces in the image
# faceRects = faceDetector(im,0)
# print("Number of faces detected:", len(faceRects))

# # list to store landmarks of all detected faces
# landmarksAll = []

# Detect landmarks


# Convert image to floating point in the range 0 to 1


# Display the results
# plt.imshow(imNorm[:,:,::-1])
# plt.title("Aligned Image")
# plt.show()



# def normalizeImagesAndLandmarks(outSize, imIn, pointsIn):
# 	# Corners of the eye in input image
# 	h, w = outSize
# 	if len(pointsIn) == 68:
# 		eyecornerSrc = [pointsIn[36], pointsIn[45]]
# 	elif len(pointsIn) == 5:
# 		eyecornerSrc = [pointsIn[2], pointsIn[0]]

# 	# Conrner of the eye in normalized image
# 	eyecornerDst = [(np.int(0.3 * w), np.int(h/3)),(np.int(.7 * w), np.int(h/3))]

# 	# Calculate similarity transform
# 	tform = fbc.similarityTransform(eyecornerSrc, eyecornerDst)
# 	imOut = np.zeros(imIn.shape, dtype=imIn.dtype)

# 	# Apply similarity transform to input image
# 	imOut = cv2.warpAffine(imIn, tform, (w, h))

# 	# Reshape pointsIn from numLandmarks x 2 to numLandmarks x 1 x 2
# 	points2 = np.reshape(pointsIn, (pointsIn.shape[0], 1, pointsIn.shape[1]))

# 	# Apply similarity transform to landmarks
# 	pointsOut = cv2.transform(points2, tform)

# 	# reshape pointsOut to numlandmarks x 2
# 	pointsOut = np.reshape(pointsOut, (pointsIn.shape[0], pointsIn.shape[1]))

#	return imOut, pointsOut

# def writeLandmarksToFile(landmarks, landmarksFileName):
# 	with open(landmarksFileName, 'w') as f:
# 		for p in landmarks.parts():
# 			f.write("%s %s\n" %(int(p.x),int(p.y)))
# 	f.close()


# S3_BUCKET = os.environ['S3_BUCKET'] if 'S3_BUCKET' in os.environ else 'mobilenetv2evas1'
# MODEL_PATH = os.environ['MODEL_PATH'] if 'MODEL_PATH' in os.environ else 'mobilenetv2.pt '

# print("Downloading model.....")

# s3 = boto3.client('s3')




# try:
#     if os.path.isfile(MODEL_PATH) !=True:

#         print("fetching object from S3")
#         obj = s3.get_object(Bucket=S3_BUCKET, Key=MODEL_PATH)
#         print("Saving it in Runtime")
#         bytestream = io.BytesIO(obj['Body'].read())
#         print("Extracting Model")
#         model=torch.jit.load(bytestream)  
# except Exception as e:
#     print(repr(e))
#     raise(e)


# def transform_image(image_bytes):
#     try:
#         transformations = transforms.Compose([
#             transforms.Resize(255),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
#         image = Image.open(io.BytesIO(image_bytes))
#         return transformations(image).unsqueeze(0)
#     except Exception as e:
#         print(repr(e))
#         raise(e)


# def get_prediction(image_bytes):
#     tensor = transform_image(image_bytes=image_bytes)
#     return model(tensor).argmax().item()


def align_face_handler(event, context):
    try:
        content_type_header = event['headers']['content-type']
        print(event['body'])
        body = base64.b64decode(event["body"])

        picture = decoder.MultipartDecoder(body, content_type_header).parts[0]
        prediction = align_face(bytes_im=picture.content)

        filename = picture.headers[b'Content-Disposition'].decode().split(';')[1].split('=')[1]
        if len(filename) < 4:
            filename = picture.headers[b'Content-Disposition'].decode().split(';')[2].split('=')[1]

        return {
            "statusCode": 200,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True
            },
            "body": json.dumps({'file': filename.replace('"', ''), 'alignedFace': img_to_base64(prediction)})
        }
    except Exception as e:
        print(repr(e))
        return {
            "statusCode": 500,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True
            },
            "body": json.dumps({"error": repr(e)})
        }


def face_swap_handler(event,context):
    try:
        content_type_header = event['headers']['content-type']
        body = base64.b64decode(event["body"])
        print('content_type header: ' + content_type_header)
        print('Event Body: ' + event["body"])
        #json_body = json.loads(event["body"])
        picture1 = decoder.MultipartDecoder(body, content_type_header).parts[0]
        picture2 = decoder.MultipartDecoder(body, content_type_header).parts[1]
        im1 = np.frombuffer(picture1.content, dtype=np.uint8)
        im2 = np.frombuffer(picture2.content, dtype=np.uint8)

        print('Body Loaded')

        #picture = decoder.MultipartDecoder(body, content_type_header).parts[0]
        #print(f'MultipartDecoder processed')
        swapped_face = fsp.face_swap(im1,im2,PREDICTOR_PATH_68)

        return {
                "statusCode": 200,
                "headers": {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                    "Access-Control-Allow-Credentials": True
                },
                "body": json.dumps({'file': 'swap', 'swapface': img_to_base64(swapped_face)})
            }
    except Exception as e:
        print(repr(e))
        return {
            "statusCode": 500,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True
            },
            "body": json.dumps({"error": repr(e)})
        }