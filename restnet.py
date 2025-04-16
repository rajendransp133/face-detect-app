import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
from controller.dbQuery import get_all_employees
from PIL import Image
import pandas as pd
from imutils.video import WebcamVideoStream
import imutils
import cv2
import time

stream_active = False
vs = None
DB_NAME = "user_db6.db"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
detected_faces = {}  
reference_embeddings = None
reference_names = None
mtcnn=None
resnet=None


def generate_reference_embeddings():
    print("Generating reference embeddings...")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Running on device: {device}')

    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, prewhiten=True,
        device=device
    )
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    
    employees = get_all_employees(DB_NAME)
    
    if not employees:
        print("No employees found in database")
        return None, None
        
    aligned_faces = []
    names = []
    
    for employee in employees:
        emp_id, name, photo_path, photo_path2,hindi_name,tamil_name,designation = employee
        
        for photo_path in [photo_path, photo_path2]:
            try:
                img = Image.open(photo_path)
                x_aligned, prob = mtcnn(img, return_prob=True)
                
                if x_aligned is not None:
                    print(f'Face detected for {name} with probability: {prob:.8f}')
                    aligned_faces.append(x_aligned)
                    names.append(name)
                else:
                    print(f'Warning: No face detected in {photo_path}')
            except Exception as e:
                print(f"Error processing {photo_path}: {e}")
    
    if not aligned_faces:
        print("No faces were detected in any of the employee photos")
        return None, None
        
    try:
        aligned_batch = torch.stack(aligned_faces).to(device)
        with torch.no_grad():
            embeddings = resnet(aligned_batch).cpu()
            
        embeddings_tensor = torch.stack([e for e in embeddings]) if isinstance(embeddings, list) else embeddings
        return embeddings, names
    except Exception as e:
        print(f"Error during embedding generation: {e}")
        return None, None

def cos_sim(a, b):
    if torch.is_tensor(a) and torch.is_tensor(b):
        a_flat = a.flatten()
        b_flat = b.flatten()
        eps = 1e-8
        return torch.dot(a_flat, b_flat) / (torch.norm(a_flat) * torch.norm(b_flat) + eps)
    else:
        if torch.is_tensor(a):
            a = a.cpu().detach().numpy()
        if torch.is_tensor(b):
            b = b.cpu().detach().numpy()
        a_flat = a.flatten()
        b_flat = b.flatten()
        norm_a = np.linalg.norm(a_flat)
        norm_b = np.linalg.norm(b_flat)
        eps = 1e-8
        if norm_a < eps or norm_b < eps:
            return 0.0
        return np.dot(a_flat, b_flat) / (norm_a * norm_b)

def cos(a, b):
    minx = -1.0
    maxx = 1.0
    sim = cos_sim(a, b)
    sim = torch.clamp(sim, minx, maxx) if torch.is_tensor(sim) else np.clip(sim, minx, maxx)
    return (sim - minx) / (maxx - minx)

def verify_faces(current_embeddings, ref_embeddings, ref_names, detected_boxes, image_to_draw, threshold=0.85):
    global detected_faces

    if detected_boxes is None or current_embeddings is None or ref_embeddings is None or ref_names is None:
        return
    
    detected_faces = {}
    
    for i, ref_emb in enumerate(ref_embeddings):
        ref_name = ref_names[i]
        for j, current_emb in enumerate(current_embeddings):
            if j < len(detected_boxes):
                dist = cos(ref_emb, current_emb)

                if dist.item() > threshold:
                    box = detected_boxes[j]
                    if box is not None:
                        x1, y1, x2, y2 = [int(coord) for coord in box]

                        display_text = f"{ref_name}"
                        text_x = x1
                        text_y = y2 + 17
                        
                        (text_width, text_height), baseline = cv2.getTextSize(
                            display_text, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 1
                        )
                        cv2.rectangle(
                            image_to_draw, 
                            (text_x, text_y - text_height - baseline), 
                            (text_x + text_width, text_y + baseline), 
                            (0, 0, 0), -1
                        )

                        cv2.putText(
                            image_to_draw, display_text, (text_x, text_y),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1, cv2.LINE_AA
                        )
                        
                        detected_faces[j] = {
                            "name": ref_name,
                            "similarity": dist.item(),
                            "box": [int(x) for x in box]
                        }

def initialize_face_recognition():
    global mtcnn, resnet, reference_embeddings, reference_names
    
    reference_embeddings, reference_names = generate_reference_embeddings()
    
    if reference_embeddings is None or reference_names is None:
        print("Failed to generate reference embeddings.")
        return False
    
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, prewhiten=True,
        device=device, keep_all=True
    )
    
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    
    return True

def process_frame(frame):
    global mtcnn, resnet, reference_embeddings, reference_names
    
    frame = cv2.flip(frame, 1) 
    frame = imutils.resize(frame, width=640)
    
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    with torch.no_grad():
        boxes, probs = mtcnn.detect(img_pil)
        img_cropped_batch = mtcnn(img_pil)
    
    frame_draw = frame.copy()
    
    if boxes is not None and img_cropped_batch is not None:
        with torch.no_grad():
            img_embedding_batch = resnet(img_cropped_batch.to(device)).cpu()
        
        for box in boxes:
            box_int = [int(b) for b in box]
            cv2.rectangle(
                frame_draw, 
                (box_int[0], box_int[1]), 
                (box_int[2], box_int[3]), 
                (0, 255, 0), 2
            )
        
        verify_faces(
            img_embedding_batch, 
            reference_embeddings, 
            reference_names, 
            boxes, 
            frame_draw, 
            threshold=0.85
        )
    
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(
        frame_draw, timestamp, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
    )
    
    return frame_draw

def get_stream_frames():
    global vs, stream_active
    
    if vs is None:
        vs = WebcamVideoStream(src=0).start()
        if vs.stream is None or not vs.stream.isOpened():
            yield (b'--frame\r\n'
                   b'Content-Type: text/plain\r\n\r\n'
                   b'Error: Could not open webcam.\r\n\r\n')
            return
    
    stream_active = True
    
    while stream_active:
        frame = vs.read()
        
        if frame is None:
            print("Warning: Failed to grab frame")
            time.sleep(0.1)
            continue
        
        try:
            frame_with_detections = process_frame(frame)
            
            ret, buffer = cv2.imencode('.jpg', frame_with_detections)
            
            if not ret:
                continue
                
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            time.sleep(0.03) 
            
        except Exception as e:
            print(f"Error in streaming: {str(e)}")
            time.sleep(0.1)
    
    if vs is not None:
        vs.stop()
        vs = None