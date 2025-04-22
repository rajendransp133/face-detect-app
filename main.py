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
import io
import shutil


from models.database import create_database
from controller.dbQuery import insert_employee, get_all_employees, get_employee, delete_employee, delete_all_employees,update_employee
from restnet import initialize_face_recognition,get_stream_frames,process_frame
from restnet import vs,stream_active,mtcnn,resnet


app = FastAPI(title="Employee Photo Manager")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

templates = Jinja2Templates(directory="templates")

DB_NAME = "test_db.db"

os.makedirs("uploads", exist_ok=True)


create_database(DB_NAME)


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return RedirectResponse(url="/viewAllUser/")

@app.get("/upload/", response_class=HTMLResponse)
async def upload_form(request: Request):
    return templates.TemplateResponse("uploadfile-c.html", {"request": request})

@app.post("/uploader/")
async def create_upload_file(
    request: Request,
    name: Annotated[str, Form()],
    files: Annotated[List[UploadFile], File()],
    designation: Annotated[str, Form()],
    hindi_name: Annotated[str, Form()] = None, 
    tamil_name: Annotated[str, Form()] = None
):
    message = ""
    try:
        if not name.strip():
            message = "Name cannot be empty"
            raise ValueError(message)
        
        if not designation.strip():
            message = "designation cannot be empty"
            raise ValueError(message)
        
        if len(files) != 2:
            message = "Please upload exactly two files"
            raise ValueError(message)

        for file in files:
            content_type = file.content_type
            if not content_type or not content_type.startswith('image/'):
                message = f"File {file.filename} is not an image"
                raise ValueError(message)

        employee_dir = os.path.join("uploads", name)
        os.makedirs(employee_dir, exist_ok=True)

        file_paths = []
        for i, file in enumerate(files, 1):
            file_extension = os.path.splitext(file.filename)[1] 
            if file_extension != ".jpeg":
                file_extension = ".jpeg"  
                
            file_path = os.path.join(employee_dir, f"{i}{file_extension}")
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            file_paths.append(file_path)

        insert_employee(DB_NAME, name, file_paths[0], file_paths[1],designation,hindi_name, tamil_name)
        message = f"Files uploaded successfully for '{name}' ✅"
        initialize_face_recognition()
    except Exception as e:
        if not message:
            message = f"Error: {str(e)}"
        for path in file_paths:
            if os.path.exists(path):
                os.remove(path)

    return templates.TemplateResponse("uploadfile-c.html", {
        "request": request,
        "message": message
    })


@app.get("/viewUser/{id}", response_class=HTMLResponse)
async def view_user(request: Request, id: int):
    employee = get_employee(DB_NAME, id)
    
    if not employee:
        raise HTTPException(status_code=404, detail=f"User with ID {id} not found")
    
    emp_id, name, photo_path, photo_path2,hindi_name ,tamil_name,designation = employee
        
    return templates.TemplateResponse(
        "showUser-c.html", 
        {
            "request": request,
            "id": emp_id,
            "name": name,
            "photopath": photo_path.replace("uploads/", ""),
            "photopath2": photo_path2.replace("uploads/", ""),
            "hindi_name":hindi_name,
            "tamil_name":tamil_name,
            "designation":designation
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
        designations=[]
        
        for employee in employees:
            emp_id, name, photo_path, photo_path2,hindi_name,tamil_name,designation = employee
            ids.append(emp_id)
            names.append(name)
            photo_paths.append(photo_path.replace("uploads/", ""))
            photo_paths2.append(photo_path2.replace("uploads/", ""))
            hindi_names.append(hindi_name)
            tamil_names.append(tamil_name)
            designations.append(designation)

            
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
                "tamil_name_list":tamil_names,
                "designation_list":designations
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
            _, _, photo_path, photo_path2,_,_,_= employee
            for path in [photo_path, photo_path2]:
                if os.path.exists(path):
                    os.remove(path)
            
            delete_employee(DB_NAME, id)
        else:
            raise HTTPException(status_code=404, detail=f"User with ID {id} not found")
        initialize_face_recognition()

        return RedirectResponse(url="/viewAllUser/", status_code=303)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting user: {str(e)}")
    
@app.get("/delAllUser/", response_class=HTMLResponse)
async def delete_all_users(request: Request):
    try:
        employees = get_all_employees(DB_NAME)
        if employees:  # Check if employees is not None and not empty
            for employee in employees:
                _, _, photo_path, photo_path2,_,_,_ = employee
                for path in [photo_path, photo_path2]:
                    if os.path.exists(path):
                        os.remove(path)
        
        delete_all_employees(DB_NAME)
        initialize_face_recognition()

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
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, name, photo_path, photo_path2, hindi_name, tamil_name,designations FROM employees WHERE name = ?",
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
                "designations": row[6] if row[6] else "-",
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


@app.post("/processFrame")
async def process_frame_route(frame: UploadFile = File(...)):
    from restnet import mtcnn, resnet, initialize_face_recognition
    
    # Initialize face recognition if not already initialized
    if mtcnn is None or resnet is None:
        success = initialize_face_recognition()
        if not success:
            raise HTTPException(status_code=500, detail="Failed to initialize face recognition")
    
    data = await frame.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    
    # Process the frame with face detection and bounding boxes
    processed = process_frame(img)
    
    # Return the processed image with bounding boxes
    _, encoded = cv2.imencode('.jpg', processed)
    return StreamingResponse(io.BytesIO(encoded.tobytes()), media_type="image/jpeg")


@app.get("/editUser/{id}", response_class=HTMLResponse)
async def edit_user_form(request: Request, id: int):
    employee = get_employee(DB_NAME, id)
    
    if not employee:
        raise HTTPException(status_code=404, detail=f"User with ID {id} not found")
    
    emp_id, name, photo_path, photo_path2, hindi_name, tamil_name, designation = employee
    
    return templates.TemplateResponse("editUser.html", {
        "request": request,
        "id": emp_id,
        "name": name,
        "photo_path": photo_path.replace("uploads/", ""),
        "photo_path2": photo_path2.replace("uploads/", ""),
        "hindi_name": hindi_name,
        "tamil_name": tamil_name,
        "designation": designation
    })

@app.put("/editUser/{id}", response_class=HTMLResponse)
async def update_user(
    request: Request,
    id: int,
    name: Annotated[str, Form()],
    designation: Annotated[str, Form()],
    hindi_name: Annotated[str, Form()] = None, 
    tamil_name: Annotated[str, Form()] = None,
    files: Annotated[List[UploadFile], File()] = []
):
    employee = get_employee(DB_NAME, id)
    
    if not employee:
        raise HTTPException(status_code=404, detail=f"User with ID {id} not found")
    
    old_id, old_name, old_photo_path, old_photo_path2, old_hindi_name, old_tamil_name, old_designation = employee
    message = ""
    file_paths = []
    
    try:
        if not name.strip():
            message = "Name cannot be empty"
            raise ValueError(message)
        
        if not designation.strip():
            message = "Designation cannot be empty"
            raise ValueError(message)
        
        # Check if files are actually uploaded (with content)
        valid_files = [file for file in files if file.filename and file.size > 0]
        
        # If any new files are provided, process them
        if valid_files:
            # Create directory for employee if it doesn't exist
            employee_dir = os.path.join("uploads", name)
            os.makedirs(employee_dir, exist_ok=True)
            
            # Track which files were successfully processed
            processed_files = []
            
            # Process each valid file
            for file in valid_files:
                # Determine which file it replaces (1 or 2)
                # For simplicity's sake, first file replaces photo 1, second file replaces photo 2
                file_index = len(processed_files) + 1
                if file_index > 2:  # We only handle 2 photos
                    break
                    
                file_extension = os.path.splitext(file.filename)[1]
                if not file_extension.lower() == ".jpeg":
                    file_extension = ".jpeg"
                    
                file_path = os.path.join(employee_dir, f"{file_index}{file_extension}")
                
                # Save the file
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                
                processed_files.append((file_index, file_path))
            
            # Initialize with old paths
            new_photo_path = old_photo_path
            new_photo_path2 = old_photo_path2
            
            # Update paths with new files
            for idx, path in processed_files:
                if idx == 1:
                    new_photo_path = path
                elif idx == 2:
                    new_photo_path2 = path
            
            # Handle name change scenario for unupdated files
            if name != old_name:
                # Move file 1 if it wasn't replaced
                if new_photo_path == old_photo_path:
                    new_dir = os.path.join("uploads", name)
                    os.makedirs(new_dir, exist_ok=True)
                    
                    file_extension = os.path.splitext(old_photo_path)[1] or ".jpeg"
                    new_path = os.path.join(new_dir, f"1{file_extension}")
                    
                    if os.path.exists(old_photo_path):
                        shutil.copy2(old_photo_path, new_path)
                        new_photo_path = new_path
                
                # Move file 2 if it wasn't replaced
                if new_photo_path2 == old_photo_path2:
                    new_dir = os.path.join("uploads", name)
                    os.makedirs(new_dir, exist_ok=True)
                    
                    file_extension = os.path.splitext(old_photo_path2)[1] or ".jpeg"
                    new_path = os.path.join(new_dir, f"2{file_extension}")
                    
                    if os.path.exists(old_photo_path2):
                        shutil.copy2(old_photo_path2, new_path)
                        new_photo_path2 = new_path
                old_dir_del = os.path.join("uploads", old_name)
                if os.path.isdir(old_dir_del):
                    shutil.rmtree(old_dir_del)  # Delete directory and its contents
                else:
                    os.remove(old_dir_del)      # Delete file
        else:
            # No new files - keep existing ones but handle name change
            if name != old_name:
                new_dir = os.path.join("uploads", name)
                os.makedirs(new_dir, exist_ok=True)
                
                # Copy file 1
                file_extension = os.path.splitext(old_photo_path)[1] or ".jpeg"
                new_photo_path = os.path.join(new_dir, f"1{file_extension}")
                
                if os.path.exists(old_photo_path):
                    shutil.copy2(old_photo_path, new_photo_path)
                else:
                    new_photo_path = old_photo_path  # Keep old path if file not found
                
                # Copy file 2
                file_extension = os.path.splitext(old_photo_path2)[1] or ".jpeg"
                new_photo_path2 = os.path.join(new_dir, f"2{file_extension}")
                
                if os.path.exists(old_photo_path2):
                    shutil.copy2(old_photo_path2, new_photo_path2)
                else:
                    new_photo_path2 = old_photo_path2  # Keep old path if file not found
                
                old_dir_del = os.path.join("uploads", old_name)
                if os.path.isdir(old_dir_del):
                    shutil.rmtree(old_dir_del)  # Delete directory and its contents
                else:
                    os.remove(old_dir_del)      # D
            else:
                # Name unchanged, keep current paths
                new_photo_path = old_photo_path
                new_photo_path2 = old_photo_path2


        # Update the employee record in the database
        update_employee(
            DB_NAME, 
            id, 
            name, 
            new_photo_path, 
            new_photo_path2, 
            designation, 
            hindi_name, 
            tamil_name
        )
        
        # Reinitialize face recognition with updated data
        initialize_face_recognition()
        
        # Clean up old directories if name changed
        if name != old_name:
            old_dir = os.path.dirname(old_photo_path)
            if os.path.exists(old_dir) and old_dir != os.path.dirname(new_photo_path):
                # Check if directory is empty before deleting
                if not os.listdir(old_dir):
                    shutil.rmtree(old_dir)
        
        message = f"Employee '{name}' updated successfully ✅"
        return RedirectResponse(url=f"/viewUser/{id}", status_code=303)
        
    except Exception as e:
        if not message:
            message = f"Error: {str(e)}"
        
        return templates.TemplateResponse("editUser.html", {
            "request": request,
            "id": id,
            "name": name,
            "hindi_name": hindi_name or old_hindi_name,
            "tamil_name": tamil_name or old_tamil_name,
            "designation": designation,
            "message": message
        })