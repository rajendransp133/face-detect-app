from fastapi import FastAPI, File, UploadFile, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse,JSONResponse
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
from controller.dbQuery import insert_employee, get_all_employees, get_employee, delete_employee, delete_all_employees
from restnet import initialize_face_recognition,get_stream_frames
from restnet import vs,stream_active,mtcnn,resnet


app = FastAPI(title="Employee Photo Manager")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

templates = Jinja2Templates(directory="templates")

DB_NAME = "user_db.db"

os.makedirs("uploads", exist_ok=True)


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
    files: Annotated[List[UploadFile], File()],
    hindi_name: Annotated[str, Form()] = None, 
    tamil_name: Annotated[str, Form()] = None
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
        insert_employee(DB_NAME, name, file_paths[0], file_paths[1],hindi_name,tamil_name)
    except Exception as e:
        for path in file_paths:
            if os.path.exists(path):
                os.remove(path)
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    return {
        "message": f"Files uploaded and stored in DB under name '{name}'",
        "filenames": [file.filename for file in files],
        "saved_paths": file_paths,
        "view_all_url": "/viewAllUser/"
    }

@app.get("/viewUser/{id}", response_class=HTMLResponse)
async def view_user(request: Request, id: int):
    employee = get_employee(DB_NAME, id)
    
    if not employee:
        raise HTTPException(status_code=404, detail=f"User with ID {id} not found")
    
    emp_id, name, photo_path, photo_path2,hindi_name ,tamil_name = employee
        
    return templates.TemplateResponse(
        "showUser-c.html", 
        {
            "request": request,
            "id": emp_id,
            "name": name,
            "photopath": photo_path.replace("uploads/", ""),
            "photopath2": photo_path2.replace("uploads/", ""),
            "hindi_name":hindi_name,
            "tamil_name":tamil_name
        }
    )

@app.get("/viewAllUser/", response_class=HTMLResponse)
async def view_all_user(request: Request):
    try:
        employees = get_all_employees(DB_NAME)
        
        ids = []
        names = []
        photo_paths = []
        photo_paths2 = []
        hindi_names=[]
        tamil_names=[]
        
        for employee in employees:
            emp_id, name, photo_path, photo_path2,hindi_name,tamil_name = employee
            ids.append(emp_id)
            names.append(name)
            photo_paths.append(photo_path.replace("uploads/", ""))
            photo_paths2.append(photo_path2.replace("uploads/", ""))
            hindi_names.append(hindi_name)
            tamil_names.append(tamil_name)

            
        return templates.TemplateResponse(
            "showUserall-c.html", 
            {
                "request": request, 
                "ids": ids,
                "name_list": names, 
                "photopath_list": photo_paths, 
                "photopath2_list": photo_paths2,
                "zip": zip,
                "hindi_name_list":hindi_names,
                "tamil_name_list":tamil_names
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

@app.get("/delUser/{id}", response_class=HTMLResponse)
async def delete_users(request: Request, id: int):
    try:
        employee = get_employee(DB_NAME, id)
        if employee:
            _, _, photo_path, photo_path2,_,_ = employee
            for path in [photo_path, photo_path2]:
                if os.path.exists(path):
                    os.remove(path)
            
            delete_employee(DB_NAME, id)
        else:
            raise HTTPException(status_code=404, detail=f"User with ID {id} not found")
        
        return RedirectResponse(url="/viewAllUser/", status_code=303)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting user: {str(e)}")
    
@app.get("/delAllUser/", response_class=HTMLResponse)
async def delete_all_users(request: Request):
    try:
        employees = get_all_employees(DB_NAME)
        if employees:  # Check if employees is not None and not empty
            for employee in employees:
                _, _, photo_path, photo_path2,_,_ = employee
                for path in [photo_path, photo_path2]:
                    if os.path.exists(path):
                        os.remove(path)
        
        delete_all_employees(DB_NAME)
        return RedirectResponse(url="/viewAllUser/", status_code=303)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting all users: {str(e)}")
    
@app.get("/detectedFaces")
async def get_detected_faces():
    from restnet import detected_faces

    # Check if any faces are detected
    if not detected_faces:
        return JSONResponse({})  # Return empty dict if no faces

    result = {}
    
    # Get face detection data - detected_faces is a dict with numeric keys
    for face_idx, face_data in detected_faces.items():
        detected_name = face_data["name"]
        conn = sqlite3.connect("user_db.db")
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, name, photo_path, photo_path2, hindi_name, tamil_name FROM employees WHERE name = ?",
            (detected_name,)
        )
        row = cursor.fetchone()
        conn.close()
        
        if row:
            # Use the database ID as the key in our results
            result[row[0]] = {
                "name": row[1],
                "hindi_name": row[4] if row[4] else "-",
                "tamil_name": row[5] if row[5] else "-",
                "similarity": face_data["similarity"]
            }
    
    return JSONResponse(result)


@app.get("/webcamStream/", response_class=HTMLResponse)
async def webcam_stream_page(request: Request):
    return templates.TemplateResponse("webcam_stream.html", {"request": request})

@app.get("/videoFeed")
async def video_feed():
    global mtcnn, resnet
    if mtcnn is None or resnet is None:
        success = initialize_face_recognition()
        if not success:
            return HTMLResponse(content="Failed to initialize face recognition system")
    
    return StreamingResponse(
        get_stream_frames(),
        media_type="multipart/x-mixed-replace;boundary=frame"
    )

@app.post("/stopStream")
async def stop_stream():
    global stream_active, vs
    stream_active = False
    if vs is not None:
        vs.stop()
        vs = None
    return {"status": "Stream stopped"}

@app.get("/detectFrame/")
async def detect_frame(request: Request):
    return RedirectResponse(url="/webcamStream/", status_code=303)
