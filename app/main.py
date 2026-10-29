from __future__ import annotations

import os
from typing import List, Optional
import numpy as np
import torch

from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db, init_db, User
from app.face_service import FaceService, NoFaceDetectedError
from app.security import require_api_key

app = FastAPI(
    title="Face Recognition & Detection System",
    description="Real-time face detection and ArcFace identity recognition powered by YOLO & PyTorch.",
    version="2.0.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/")
async def read_index():
    return FileResponse("app/static/index.html")


# Initialize Database
init_db()

# Initialize Face Service
face_service = FaceService()


@app.get("/status")
async def get_system_status(db: Session = Depends(get_db)):
    """System diagnostic and status endpoint."""
    user_count = 0
    try:
        user_count = db.query(User).count()
    except Exception:
        pass

    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else None

    return {
        "status": "online",
        "device": str(face_service.device),
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "enrolled_count": user_count,
        "models": {
            "yolo": os.path.basename(settings.yolo_model_path),
            "arcface": os.path.basename(settings.arcface_ckpt_path),
        },
    }


@app.get("/api/verify-key", dependencies=[Depends(require_api_key)])
async def verify_api_key():
    """Verify that the supplied API key is valid."""
    return {"valid": True, "message": "API key verified successfully"}


@app.get("/users", dependencies=[Depends(require_api_key)])
async def list_users(db: Session = Depends(get_db)):
    """List all enrolled user profiles."""
    users = db.query(User).order_by(User.id.desc()).all()
    return [{"id": u.id, "name": u.name} for u in users]


@app.delete("/users/{user_id}", dependencies=[Depends(require_api_key)])
async def delete_user(user_id: int, db: Session = Depends(get_db)):
    """Delete an enrolled user profile."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail=f"User with ID {user_id} not found")

    user_name = user.name
    db.delete(user)
    db.commit()
    return {"message": f"Successfully deleted '{user_name}'", "id": user_id}


@app.post("/enroll", dependencies=[Depends(require_api_key)])
async def enroll_user(
    name: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Enroll a new identity into the database."""
    name = name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Name cannot be empty")

    try:
        image_bytes = await file.read()
        embedding, box, det_conf = face_service.get_embedding(image_bytes)

        # Save to DB
        new_user = User(name=name, embedding=embedding)
        db.add(new_user)
        db.commit()
        db.refresh(new_user)

        return {
            "message": f"User '{name}' enrolled successfully",
            "user_id": new_user.id,
            "box": box,
            "det_conf": float(det_conf),
        }
    except NoFaceDetectedError as e:
        raise HTTPException(
            status_code=400,
            detail="No face detected in the image. Please make sure your face is clearly visible and centered.",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@app.post("/recognize", dependencies=[Depends(require_api_key)])
async def recognize_user(
    file: UploadFile = File(...),
    threshold: float = Query(0.5, ge=0.0, le=1.0),
    db: Session = Depends(get_db),
):
    """Match a face against enrolled identities."""
    try:
        image_bytes = await file.read()
        target_embedding, box, det_conf = face_service.get_embedding(image_bytes)

        users = db.query(User).all()
        if not users:
            return {
                "match": "Unknown",
                "score": 0.0,
                "box": box,
                "det_conf": float(det_conf),
                "message": "No enrolled identities in database",
            }

        best_score = -1.0
        best_match = "Unknown"

        for user in users:
            # Cosine similarity
            user_emb = np.asarray(user.embedding, dtype=np.float32)
            score = float(np.dot(target_embedding, user_emb))

            if score > best_score:
                best_score = score
                best_match = user.name

        if best_score < threshold:
            best_match = "Unknown"

        return {
            "match": best_match,
            "score": best_score,
            "box": box,
            "det_conf": float(det_conf),
            "message": "Face recognized" if best_match != "Unknown" else "Unrecognized person",
        }

    except NoFaceDetectedError:
        # For real-time camera stream, return 200 with no face detected
        return {
            "match": None,
            "score": 0.0,
            "box": None,
            "det_conf": 0.0,
            "message": "No face detected in frame",
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recognition failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
