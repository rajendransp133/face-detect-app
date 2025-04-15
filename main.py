
from fastapi import FastAPI, File, UploadFile, Request, Form, HTTPException
import uvicorn
import shutil
from fastapi.responses import HTMLResponse, RedirectResponse
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

# Configure application
app = FastAPI(title="Employee Photo Manager")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

templates = Jinja2Templates(directory="templates")

DB_NAME = "employee_data.db"

# Ensure directories exist
os.makedirs("uploads", exist_ok=True)

# Create database and tables if they don't exist
def create_database(db_name):
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        print("✅ Successfully connected to SQLite")
        create_table_query = '''
            CREATE TABLE IF NOT EXISTS employees (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                photo_path TEXT NOT NULL,
                photo_path2 TEXT NOT NULL
            );
        '''
        cursor.execute(create_table_query)
        conn.commit()
        print("📁 SQLite table 'employees' created (if it didn't already exist)")

    except sqlite3.Error as error:
        print("❌ Error while creating a SQLite table:", error)

    finally:
        if conn:
            cursor.close()
            conn.close()
            print("🔌 SQLite connection is closed")

# Initialize database
create_database(DB_NAME)

# Database Operations
def insert_employee(db_name, name, photo_path, photo_path2):
    conn = None
    
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        
        # Insert paths rather than blobs
        insert_query = """INSERT INTO employees
                          (name, photo_path, photo_path2) VALUES (?, ?, ?)"""
        
        data_tuple = (name, photo_path, photo_path2)
        cursor.execute(insert_query, data_tuple)
        conn.commit()
        cursor.close()
        return True

    except sqlite3.Error as error:
        print("Failed to insert employee data:", error)
        raise
    finally:
        if conn:
            conn.close()

def get_all_employees(db_name):
    conn = None
    employees = []
    
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        query = "SELECT id, name, photo_path, photo_path2 FROM employees"
        cursor.execute(query)
        employees = cursor.fetchall()
        
        cursor.close()
        
    except sqlite3.Error as error:
        print("Failed to read data from sqlite table", error)
    finally:
        if conn:
            conn.close()
            
    return employees

def get_employee(db_name, emp_id):
    conn = None
    employee = None
    
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        query = "SELECT id, name, photo_path, photo_path2 FROM employees WHERE id = ?"
        cursor.execute(query, (emp_id,))
        employee = cursor.fetchone()
        
        cursor.close()
        
    except sqlite3.Error as error:
        print("Failed to read data from sqlite table", error)
    finally:
        if conn:
            conn.close()
            
    return employee

def delete_employee(db_name, emp_id):
    conn = None

    try:
        emp_id = int(emp_id)
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        # Delete the record
        cursor.execute("DELETE FROM employees WHERE id = ?", (emp_id,))
        conn.commit()
        cursor.close()  

        return True

    except sqlite3.Error as error:
        print("Error while deleting user:", error)
        return False
    finally:
        if conn:
            conn.close()

def delete_all_employees(db_name):
    conn = None

    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        # Delete all entries
        cursor.execute("DELETE FROM employees")
        conn.commit()
        cursor.close()

        return True

    except sqlite3.Error as error:
        print("Error while deleting all users:", error)
        return False
    finally:
        if conn:
            conn.close()

# Face Recognition Functions
def cos_sim(a, b):
    """Calculates cosine similarity between two vectors."""
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
    """Calculates scaled cosine similarity [0, 1]."""
    minx = -1.0
    maxx = 1.0
    sim = cos_sim(a, b)
    sim = torch.clamp(sim, minx, maxx) if torch.is_tensor(sim) else np.clip(sim, minx, maxx)
    return (sim - minx) / (maxx - minx)

def generate_reference_embeddings():
    """Loads known faces from the uploads directory and generates reference embeddings."""
    print("Generating reference embeddings...")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Running on device: {device}')

    # Initialize face detection and embedding models
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, prewhiten=True,
        device=device
    )
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    
    # Get all employees from database
    employees = get_all_employees(DB_NAME)
    
    if not employees:
        print("No employees found in database")
        return None, None
        
    aligned_faces = []
    names = []
    
    for employee in employees:
        emp_id, name, photo_path, photo_path2 = employee
        
        # Process both photos for each employee
        for photo_path in [photo_path, photo_path2]:
            full_path = photo_path
            
            try:
                img = Image.open(full_path)
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
        
    # Generate embeddings
    try:
        aligned_batch = torch.stack(aligned_faces).to(device)
        with torch.no_grad():
            embeddings = resnet(aligned_batch).cpu()
            
        # Optional: Print similarity matrix
        embeddings_tensor = torch.stack([e for e in embeddings]) if isinstance(embeddings, list) else embeddings
        dists = [[cos(e1, e2).item() for e2 in embeddings_tensor] for e1 in embeddings_tensor]
        print("\nReference Embedding Similarity Matrix:")
        print(pd.DataFrame(dists, columns=names, index=names))
        
        return embeddings, names
    except Exception as e:
        print(f"Error during embedding generation: {e}")
        return None, None

def verify_faces(current_embeddings, ref_embeddings, ref_names, detected_boxes, image_to_draw, threshold=0.85):
    """Compares current embeddings with references and draws names on the image."""
    if detected_boxes is None or current_embeddings is None or ref_embeddings is None or ref_names is None:
        return

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

                        print(f"Match found: {ref_name} (Similarity: {dist.item():.2f})")

def run_face_recognition():
    """Runs the main webcam face recognition loop."""
    # Generate reference embeddings first
    reference_embeddings, reference_names = generate_reference_embeddings()

    if reference_embeddings is None or reference_names is None:
        print("Exiting due to failure in reference embedding generation.")
        return

    # Initialize models for real-time detection
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Running real-time detection on device: {device}')

    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, prewhiten=True,
        device=device, keep_all=True
    )

    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    # Start webcam stream
    vs = WebcamVideoStream(src=0).start()
    if vs.stream is None or not vs.stream.isOpened():
        print("Error: Could not open webcam.")
        return
    print("Camera on. Press 'Enter' to exit.")
    
    # Main loop
    while True:
        # Read frame
        im = vs.read()
        if im is None:
            print("Warning: Failed to grab frame from webcam.")
            continue

        # Flip horizontally
        im = cv2.flip(im, 1)

        # Resize for faster processing
        frame = imutils.resize(im, width=600)

        # Convert BGR to RGB for MTCNN
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Face detection and alignment
        with torch.no_grad():
            boxes, probs = mtcnn.detect(img_pil)
            img_cropped_batch = mtcnn(img_pil)

        # Make a copy to draw on
        frame_draw = frame.copy()

        # Verification and drawing
        if boxes is not None and img_cropped_batch is not None:
            with torch.no_grad():
                img_embedding_batch = resnet(img_cropped_batch.to(device)).cpu()

            # Draw bounding boxes
            for box in boxes:
                box_int = [int(b) for b in box]
                cv2.rectangle(
                    frame_draw, 
                    (box_int[0], box_int[1]), 
                    (box_int[2], box_int[3]), 
                    (0, 255, 0), 2
                )

            # Perform verification
            verify_faces(
                img_embedding_batch, 
                reference_embeddings, 
                reference_names, 
                boxes, 
                frame_draw, 
                threshold=0.85
            )

        # Display the result
        cv2.imshow('Face Recognition - Press Enter to Exit', frame_draw)

        # Check for exit keys
        key = cv2.waitKey(1) & 0xFF
        if key == 13 or key == ord('q'):
            print("Exit key pressed.")
            break

    # Cleanup
    print("Cleaning up...")
    cv2.destroyAllWindows()
    vs.stop()
    print("Webcam stopped.")

# FastAPI Routes
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    # Redirect to upload form
    return RedirectResponse(url="/upload/")

@app.get("/upload/", response_class=HTMLResponse)
async def upload_form(request: Request):
    return templates.TemplateResponse("uploadfile-c.html", {"request": request})

@app.post("/uploader/")
async def create_upload_file(
    name: Annotated[str, Form()],
    files: Annotated[List[UploadFile], File()]
):
    # Validate input
    if not name.strip():
        raise HTTPException(status_code=400, detail="Name cannot be empty")
    
    if len(files) != 2:
        raise HTTPException(status_code=400, detail="Please upload exactly two files")
    
    # Validate file types
    for file in files:
        content_type = file.content_type
        if not content_type or not content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail=f"File {file.filename} is not an image")
    
    # Create folder structure for this employee
    employee_dir = os.path.join("uploads", name)
    os.makedirs(employee_dir, exist_ok=True)
    
    file_paths = []
    
    # Save uploaded files into the employee's folder
    for i, file in enumerate(files, 1):
        file_extension = os.path.splitext(file.filename)[1]  # Get original extension
        if file_extension!=".jpeg":
            file_extension = ".jpeg"  # Default to .jpg if no extension
            
        file_path = os.path.join(employee_dir, f"{i}{file_extension}")
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        file_paths.append(file_path)

    # Store in database
    try:
        insert_employee(DB_NAME, name, file_paths[0], file_paths[1])
    except Exception as e:
        # Delete uploaded files if database insert fails
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
        
        # Prepare data for template
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

@app.get("/viewuser/{id}", response_class=HTMLResponse)
async def view_user(request: Request, id: int):
    employee = get_employee(DB_NAME, id)

    if not employee:
        raise HTTPException(status_code=404, detail=f"User with ID {id} not found")

    emp_id, name, photo_path, photo_path2 = employee

    # Extract the employee folder name
    folder_name = os.path.basename(os.path.dirname(photo_path))

    return templates.TemplateResponse(
        "showUser-c.html",
        {
            "request": request,
            "id": emp_id,
            "name": name,
            "name_val": folder_name,
            "photopath": os.path.basename(photo_path),
            "photopath2": os.path.basename(photo_path2),
        }
    )

@app.get("/deluserall/", response_class=HTMLResponse)
async def delete_all_users_route(request: Request):
    try:
        # Get all employees to delete their files
        employees = get_all_employees(DB_NAME)
        for employee in employees:
            _, _, photo_path, photo_path2 = employee
            for path in [photo_path, photo_path2]:
                if os.path.exists(path):
                    os.remove(path)
        
        # Then delete all records
        delete_all_employees(DB_NAME)
        return RedirectResponse(url="/viewuserall/", status_code=303)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting all users: {str(e)}")
    
@app.get("/detect-frame/")
async def detectframe(request: Request):
    run_face_recognition()
    return {"status": "Face recognition session completed"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)