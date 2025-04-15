from fastapi import FastAPI, File, UploadFile, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
import uvicorn
import shutil
from fastapi.templating import Jinja2Templates
import os
from typing import Annotated, List
from fastapi.staticfiles import StaticFiles
import sqlite3
import cv2
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
import pandas as pd
import imutils
from imutils.video import WebcamVideoStream
import threading
import time

from models.database import create_database
from controller.db_query import insert_employee, get_all_employees, get_employee, delete_employee, delete_all_employees

app = FastAPI(title="Employee Photo Manager")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

templates = Jinja2Templates(directory="templates")

DB_NAME = "employee_data.db"

os.makedirs("uploads", exist_ok=True)

# Global variables for face recognition
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = None
resnet = None
reference_embeddings = None
reference_names = None
stream_active = False
vs = None
detected_faces = {}  # Dictionary to store detected faces

create_database(DB_NAME)

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
        emp_id, name, photo_path, photo_path2 = employee
        
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
    
    # Clear previous detections
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
                        
                        # Store detected face information
                        detected_faces[j] = {
                            "name": ref_name,
                            "similarity": dist.item(),
                            "box": [int(x) for x in box]
                        }

def initialize_face_recognition():
    global mtcnn, resnet, reference_embeddings, reference_names
    
    # Generate reference embeddings
    reference_embeddings, reference_names = generate_reference_embeddings()
    
    if reference_embeddings is None or reference_names is None:
        print("Failed to generate reference embeddings.")
        return False
    
    # Initialize models
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, prewhiten=True,
        device=device, keep_all=True
    )
    
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    
    return True

def process_frame(frame):
    global mtcnn, resnet, reference_embeddings, reference_names
    
    frame = cv2.flip(frame, 1)  # Mirror the frame
    frame = imutils.resize(frame, width=640)  # Resize for web display
    
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    with torch.no_grad():
        boxes, probs = mtcnn.detect(img_pil)
        img_cropped_batch = mtcnn(img_pil)
    
    frame_draw = frame.copy()
    
    if boxes is not None and img_cropped_batch is not None:
        with torch.no_grad():
            img_embedding_batch = resnet(img_cropped_batch.to(device)).cpu()
        
        # Draw face boxes
        for box in boxes:
            box_int = [int(b) for b in box]
            cv2.rectangle(
                frame_draw, 
                (box_int[0], box_int[1]), 
                (box_int[2], box_int[3]), 
                (0, 255, 0), 2
            )
        
        # Verify faces against reference embeddings
        verify_faces(
            img_embedding_batch, 
            reference_embeddings, 
            reference_names, 
            boxes, 
            frame_draw, 
            threshold=0.85
        )
    
    # Add timestamp
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(
        frame_draw, timestamp, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
    )
    
    return frame_draw

def get_stream_frames():
    global vs, stream_active
    
    # Initialize video stream
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
            # Process the frame for face recognition
            frame_with_detections = process_frame(frame)
            
            # Convert to JPEG for streaming
            ret, buffer = cv2.imencode('.jpg', frame_with_detections)
            
            if not ret:
                continue
                
            # Yield the frame in MJPEG format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            # Control frame rate to reduce CPU usage
            time.sleep(0.03)  # ~30 FPS
            
        except Exception as e:
            print(f"Error in streaming: {str(e)}")
            time.sleep(0.1)
    
    # Clean up when streaming stops
    if vs is not None:
        vs.stop()
        vs = None

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return RedirectResponse(url="/upload/")

@app.get("/upload/", response_class=HTMLResponse)
async def upload_form(request: Request):
    return templates.TemplateResponse("uploadfile-c.html", {"request": request})

@app.post("/uploader/")
async def create_upload_file(
    name: Annotated[str, Form()],
    files: Annotated[List[UploadFile], File()]
):
    if not name.strip():
        raise HTTPException(status_code=400, detail="Name cannot be empty")
    
    if len(files) != 2:
        raise HTTPException(status_code=400, detail="Please upload exactly two files")
    
    for file in files:
        content_type = file.content_type
        if not content_type or not content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail=f"File {file.filename} is not an image")
    
    employee_dir = os.path.join("uploads", name)
    os.makedirs(employee_dir, exist_ok=True)
    
    file_paths = []
    
    for i, file in enumerate(files, 1):
        file_extension = os.path.splitext(file.filename)[1] 
        if file_extension!=".jpeg":
            file_extension = ".jpeg"  
            
        file_path = os.path.join(employee_dir, f"{i}{file_extension}")
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        file_paths.append(file_path)

    try:
        insert_employee(DB_NAME, name, file_paths[0], file_paths[1])
    except Exception as e:
        for path in file_paths:
            if os.path.exists(path):
                os.remove(path)
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    return {
        "message": f"Files uploaded and stored in DB under name '{name}'",
        "filenames": [file.filename for file in files],
        "saved_paths": file_paths,
        "view_all_url": "/viewuserall/"
    }

@app.get("/viewuser/{id}", response_class=HTMLResponse)
async def view_user(request: Request, id: int):
    employee = get_employee(DB_NAME, id)
    
    if not employee:
        raise HTTPException(status_code=404, detail=f"User with ID {id} not found")
    
    emp_id, name, photo_path, photo_path2 = employee
        
    return templates.TemplateResponse(
        "showUser-c.html", 
        {
            "request": request,
            "id": emp_id,
            "name": name,
            "photopath": photo_path.replace("uploads/", ""),
            "photopath2": photo_path2.replace("uploads/", "")
        }
    )

@app.get("/viewuserall/", response_class=HTMLResponse)
async def view_user_all(request: Request):
    try:
        employees = get_all_employees(DB_NAME)
        
        ids = []
        names = []
        photo_paths = []
        photo_paths2 = []
        
        for employee in employees:
            emp_id, name, photo_path, photo_path2 = employee
            ids.append(emp_id)
            names.append(name)
            photo_paths.append(photo_path.replace("uploads/", ""))
            photo_paths2.append(photo_path2.replace("uploads/", ""))
            
        return templates.TemplateResponse(
            "showUserall-c.html", 
            {
                "request": request, 
                "ids": ids,
                "name_list": names, 
                "photopath_list": photo_paths, 
                "photopath2_list": photo_paths2,
                "zip": zip
            }
        )
    except Exception as e:
        print(f"Error in viewuserall: {str(e)}")
        return HTMLResponse(
            content=f"""
            <html>
                <head><title>Error</title></head>
                <body>
                    <h1>Error Loading Data</h1>
                    <p>There was a problem retrieving the user data: {str(e)}</p>
                    <p><a href="/upload/">Return to upload form</a></p>
                </body>
            </html>
            """,
            status_code=500
        )

@app.get("/deluserall/", response_class=HTMLResponse)
async def delete_all_users_route(request: Request):
    try:
        employees = get_all_employees(DB_NAME)
        for employee in employees:
            _, _, photo_path, photo_path2 = employee
            for path in [photo_path, photo_path2]:
                if os.path.exists(path):
                    os.remove(path)
        
        delete_all_employees(DB_NAME)
        return RedirectResponse(url="/viewuserall/", status_code=303)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting all users: {str(e)}")

# API endpoint to get current detected faces
@app.get("/detected-faces")
async def get_detected_faces():
    return detected_faces

# New endpoints for webcam streaming
@app.get("/webcam-stream/", response_class=HTMLResponse)
async def webcam_stream_page(request: Request):
    return templates.TemplateResponse("webcam_stream.html", {"request": request})

@app.get("/video-feed")
async def video_feed():
    # Initialize face recognition system if not already done
    global mtcnn, resnet
    if mtcnn is None or resnet is None:
        success = initialize_face_recognition()
        if not success:
            return HTMLResponse(content="Failed to initialize face recognition system")
    
    return StreamingResponse(
        get_stream_frames(),
        media_type="multipart/x-mixed-replace;boundary=frame"
    )

@app.post("/stop-stream")
async def stop_stream():
    global stream_active, vs
    stream_active = False
    if vs is not None:
        vs.stop()
        vs = None
    return {"status": "Stream stopped"}

@app.get("/detect-frame/")
async def detectframe(request: Request):
    # Redirect to the webcam stream page instead of opening a popup
    return RedirectResponse(url="/webcam-stream/", status_code=303)

