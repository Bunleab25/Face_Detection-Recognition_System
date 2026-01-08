from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
import numpy as np
from typing import List

from app.database import get_db, init_db, User
from app.face_service import FaceService

app = FastAPI(title="Face Recognition System")

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse('app/static/index.html')

# Initialize Database
init_db()

# Initialize Face Service
# Uses env vars: YOLO_MODEL_PATH / ARCFACE_CKPT_PATH (see app/config.py)
face_service = FaceService()

@app.post("/enroll")
async def enroll_user(
    name: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    try:
        image_bytes = await file.read()
        embedding, _, _ = face_service.get_embedding(image_bytes)
        
        # Save to DB
        new_user = User(name=name, embedding=embedding)
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        return {"message": f"User {name} enrolled successfully", "user_id": new_user.id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recognize")
async def recognize_user(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    try:
        image_bytes = await file.read()
        target_embedding, box, det_conf = face_service.get_embedding(image_bytes)
        
        # Fetch all users
        users = db.query(User).all()
        
        if not users:
            return {"message": "No users in database", "match": None, "box": box}

        best_score = -1.0
        best_match = "Unknown"
        
        for user in users:
            # Cosine Similarity
            # embedding is stored as numpy array in PickleType
            user_emb = np.asarray(user.embedding, dtype=np.float32)
            score = float(np.dot(target_embedding, user_emb))
            
            if score > best_score:
                best_score = score
                best_match = user.name
        
        # Threshold check
        THRESHOLD = 0.5
        if best_score < THRESHOLD:
            best_match = "Unknown"
            
        return {
            "match": best_match,
            "score": best_score,
            "box": box,
            "det_conf": det_conf,
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
