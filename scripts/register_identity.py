"""Register new identities to the face database."""

import sys
from pathlib import Path

import cv2
from face_alignment import FaceAligner


def register_from_webcam(output_dir, name, num_photos=5):
    """Capture photos from webcam and register new identity.
    
    Args:
        output_dir: Base directory for identity database.
        name: Name of the person to register.
        num_photos: Number of photos to capture.
    """
    output_dir = Path(output_dir)
    identity_dir = output_dir / name.replace(' ', '_')
    identity_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize face aligner
    print("Initializing camera and face detector...")
    aligner = FaceAligner(
        yolo_model_path='models/yolo11n.pt',
        device='cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu',
        conf_threshold=0.5,
    )
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print(f"\n{'='*60}")
    print(f"REGISTERING: {name}")
    print(f"{'='*60}")
    print(f"Will capture {num_photos} photos")
    print("Press SPACE to capture a photo")
    print("Press 'q' to quit")
    print(f"{'='*60}\n")
    
    captured = 0
    
    while captured < num_photos:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect faces
        bboxes = aligner.detect_faces(frame)
        
        # Draw boxes
        display = frame.copy()
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Show instructions
        cv2.putText(display, f"Captured: {captured}/{num_photos}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display, "Press SPACE to capture", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Registration', display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # Space to capture
            if len(bboxes) == 0:
                print("⚠️  No face detected! Please position your face in frame.")
                continue
            
            if len(bboxes) > 1:
                print("⚠️  Multiple faces detected! Please ensure only one person is in frame.")
                continue
            
            # Align and save
            bbox = bboxes[0]
            aligned_face = aligner.align_face(frame, bbox)
            
            if aligned_face is not None:
                output_path = identity_dir / f"{name}_{captured+1:03d}.jpg"
                cv2.imwrite(str(output_path), aligned_face)
                captured += 1
                print(f"✓ Captured {captured}/{num_photos}: {output_path.name}")
            else:
                print("⚠️  Failed to align face. Please try again.")
        
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    if captured == num_photos:
        print(f"\n✅ Successfully registered {name} with {num_photos} photos!")
        print(f"📁 Saved to: {identity_dir}")
    else:
        print(f"\n⚠️  Only captured {captured}/{num_photos} photos.")
    
    return captured == num_photos


def register_from_image(output_dir, name, image_path):
    """Register identity from existing image file.
    
    Args:
        output_dir: Base directory for identity database.
        name: Name of the person to register.
        image_path: Path to image file.
    """
    output_dir = Path(output_dir)
    identity_dir = output_dir / name.replace(' ', '_')
    identity_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize face aligner
    aligner = FaceAligner(
        yolo_model_path='models/yolo11n.pt',
        device='cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu',
        conf_threshold=0.5,
    )
    
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Could not load image: {image_path}")
        return False
    
    # Detect faces
    bboxes = aligner.detect_faces(img)
    
    if len(bboxes) == 0:
        print("Error: No face detected in image")
        return False
    
    if len(bboxes) > 1:
        print(f"Warning: Multiple faces detected ({len(bboxes)}). Using first face.")
    
    # Align and save
    bbox = bboxes[0]
    aligned_face = aligner.align_face(img, bbox)
    
    if aligned_face is not None:
        output_path = identity_dir / f"{name}_001.jpg"
        cv2.imwrite(str(output_path), aligned_face)
        print(f"✅ Successfully registered {name}")
        print(f"📁 Saved to: {output_path}")
        return True
    else:
        print("Error: Failed to align face")
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Register new face identity')
    parser.add_argument('--name', type=str, required=True,
                       help='Name of person to register')
    parser.add_argument('--database', type=str, default='face_database',
                       help='Path to identity database directory')
    parser.add_argument('--webcam', action='store_true',
                       help='Capture from webcam (default)')
    parser.add_argument('--image', type=str,
                       help='Register from existing image file')
    parser.add_argument('--num-photos', type=int, default=5,
                       help='Number of photos to capture from webcam')
    
    args = parser.parse_args()
    
    if args.image:
        register_from_image(args.database, args.name, args.image)
    else:
        register_from_webcam(args.database, args.name, args.num_photos)


if __name__ == "__main__":
    main()
