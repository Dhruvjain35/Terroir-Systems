from ultralytics import YOLO

def main():
    print("🚀 Initializing Terroir AI - YOLOv8 Engine (M4 GPU with AMP Disabled)...")
    model = YOLO('yolov8n.pt') 
    
    results = model.train(
        data='potato_dataset/data.yaml', 
        epochs=25,       
        imgsz=640,
        batch=8,         # Lower batch size keeps the GPU memory perfectly aligned
        device='mps',    # We are back on the M4 GPU!
        amp=False,       # 🚨 THE MAGIC BULLET: Stops the Apple Silicon crash
        plots=True      
    )
    print("✅ Training Complete! Model saved as 'best.pt'")

if __name__ == '__main__':
    main()