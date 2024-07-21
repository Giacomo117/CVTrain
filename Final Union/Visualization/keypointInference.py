import torch
import numpy as np
import cv2
import torch.nn as nn

from facenet_pytorch import MTCNN
from torchvision.models import resnet50


# Crop the detected face with padding and return it
def crop_image(box, image, pre_padding=7, post_padding=7):
    x1, y1, x2, y2 = box
    x1 = x1 - pre_padding
    y1 = y1 - pre_padding
    x2 = x2 + post_padding
    y2 = y2 + post_padding
    cropped_image = image[int(y1):int(y2), int(x1):int(x2)]
    return cropped_image, x1, y1


# Reuploading the model as used in training
model = resnet50(weights='DEFAULT')
for param in model.parameters():
    param.requires_grad = True

# Change the final layer
model.fc = nn.Linear(in_features=2048, out_features=136)

model = model.to('cpu')

# Load the model checkpoint
checkpoint = torch.load(r"my_output\model.pth",
                        map_location=torch.device('cpu'))
# Load model weights state_dict
model.load_state_dict(checkpoint['model_state_dict'])

# Execute model evaluation
model.eval()

# create the MTCNN model, `keep_all=True` returns all the detected faces
mtcnn = MTCNN(keep_all=True, device='cpu')

# capture the webcam
cap = cv2.VideoCapture(0)  # Usa la webcam come sorgente video
if (cap.isOpened() == False):
    print('Error while trying to open webcam. Please check again...')

# get the frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
frame_fps = int(cap.get(5))

frame_count = 0  # To count total frames.
total_fps = 0  # To get the final frames per second.

while (cap.isOpened()):
    # capture each frame of the video
    ret, frame = cap.read()
    if ret == True:
        with torch.no_grad():
            # Invoke face detection
            bounding_boxes, conf = mtcnn.detect(frame, landmarks=False)

            # Detect keypoints if face is detected.
            if bounding_boxes is not None:
                for box in bounding_boxes:
                    # crop image around the face recognized
                    cropped_image, x1, y1 = crop_image(box, frame)
                    image = cropped_image.copy()

                    if image.shape[0] > 1 and image.shape[1] > 1:
                        # Execute preprocesing to match model's
                        image = cv2.resize(image, (224, 224))
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        image = image / 255.0
                        image = np.transpose(image, (2, 0, 1))
                        image = torch.tensor(image, dtype=torch.float)
                        image = image.unsqueeze(0).to('cpu')

                        # Execute predictions
                        outputs = model(image)

                        outputs = outputs.cpu().detach().numpy()

                        # Reshape the output from flattened to (68, 2)
                        outputs = outputs.reshape(-1, 2)
                        keypoints = outputs

                        # Draw keypoints on face
                        for i, p in enumerate(keypoints):
                            p[0] = p[0] / 224 * cropped_image.shape[1]
                            p[1] = p[1] / 224 * cropped_image.shape[0]

                            p[0] += x1
                            p[1] += y1
                            cv2.circle(
                                frame,
                                (int(p[0]), int(p[1])),
                                2,
                                (0, 0, 255),
                                -1,
                                cv2.LINE_AA
                            )

            cv2.imshow('Facial Keypoint Frame', frame)

            # press `q` to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Cambiato a 1 per aggiornare in tempo reale
                break

    else:
        break

# release VideoCapture()
cap.release()

# close all frames and video windows
cv2.destroyAllWindows()
