import argparse
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from pathlib import Path

def load_model(model_name="RavenOnur/Sign-Language"):
    """
    Load the sign language translation model and processor.
    
    Args:
        model_name (str): Name of the model on Hugging Face Hub
        
    Returns:
        tuple: (processor, model) for image processing and classification
    """
    print(f"Loading model: {model_name}")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    
    return processor, model

def classify_image(image_path, processor, model):
    """
    Classify an image containing sign language.
    
    Args:
        image_path (str): Path to the image file
        processor: Image processor for the model
        model: Classification model
        
    Returns:
        dict: Classification results with probabilities
    """
    # Load and process the image
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    
    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Get predicted class and probabilities
    predicted_class_idx = logits.argmax(-1).item()
    probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
    
    # Format results
    results = {
        "predicted_class": model.config.id2label[predicted_class_idx],
        "confidence": probabilities[predicted_class_idx].item(),
        "top_predictions": []
    }
    
    # Get top 5 predictions
    top_indices = torch.topk(probabilities, min(5, len(model.config.id2label))).indices
    for idx in top_indices:
        label = model.config.id2label[idx.item()]
        confidence = probabilities[idx].item()
        results["top_predictions"].append((label, confidence))
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Sign Language Translation using Vision Transformer")
    parser.add_argument("--image", type=str, required=True, help="Path to image containing sign language")
    parser.add_argument("--model", type=str, default="RavenOnur/Sign-Language", 
                        help="Model name on Hugging Face Hub (default: RavenOnur/Sign-Language)")
    
    args = parser.parse_args()
    
    # Validate image path
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image file '{args.image}' not found")
        return
    
    # Load model and processor
    try:
        processor, model = load_model(args.model)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Classify the image
    try:
        results = classify_image(args.image, processor, model)
        
        # Print results
        print("\nSign Language Translation Results:")
        print(f"Predicted sign: {results['predicted_class']} (Confidence: {results['confidence']:.2%})")
        
        print("\nTop predictions:")
        for i, (label, conf) in enumerate(results['top_predictions'], 1):
            print(f"{i}. {label}: {conf:.2%}")
            
    except Exception as e:
        print(f"Error classifying image: {e}")

if __name__ == "__main__":
    main()