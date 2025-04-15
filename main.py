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
from restnet import initialize_face_recognition,get_stream_frames
from restnet import vs,stream_active


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

create_database(DB_NAME)


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


@app.get("/detected-faces")
async def get_detected_faces():
    from restnet import detected_faces
    return dict(detected_faces)

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

