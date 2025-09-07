#!/usr/bin/env python3

import argparse
import torch
from PIL import Image
import numpy as np
from transformers import AutoProcessor, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from database import DetectionDatabase
import cv2
from PIL import Image


class CaptionGenerator:
    def __init__(self, db_path: str = "detections.db"):
        self.db = DetectionDatabase(db_path)
        
        # Load Microsoft GIT model for image captioning with TextVQA capabilities
        print("Loading Microsoft GIT TextVQA captioning model...")
        self.processor = AutoProcessor.from_pretrained("microsoft/git-base-textvqa", use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-textvqa")
        
        # Load sentence transformer for embeddings
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Move to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.embedding_model.to(self.device)

        print(f"Models loaded on device: {self.device}")
    
    def get_object_specific_question(self, object_type: str) -> str:
        """Generate object-specific questions for detailed captioning."""
        object_type_lower = object_type.lower()
        
        if object_type_lower in ['car', 'truck', 'bus', 'vehicle']:
            return ("What is the color, make, model, and type of this car? "
                   "Is it towing or carrying anything? Describe any visible details.")
        
        elif object_type_lower in ['person', 'people']:
            return ("What is the person's approximate age and sex? "
                   "What color hair and clothing are they wearing? "
                   "Do they have a backpack, hat, or other accessories? "
                   "What are they carrying or holding?")
        
        elif object_type_lower in ['bicycle', 'bike']:
            return ("What color and type of bicycle is this? "
                   "Does it have baskets, bags, or any accessories?")
        
        elif object_type_lower in ['motorcycle', 'motorbike']:
            return ("What color and type of motorcycle is this? "
                   "Does it have any cargo or accessories?")
        
        elif object_type_lower in ['cat', 'dog', 'bird']:
            return f"What color and breed is this {object_type_lower}? Describe its appearance and any visible features."
        
        else:
            return f"Describe this {object_type} in detail, including color, size, and any distinctive features."

    
    def generate_detailed_caption(self, image: Image.Image, object_type: str) -> str:
        """Generate detailed caption using git-base-textvqa with object-specific questions."""

        image = image.convert("RGB")
    
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
        question = self.get_object_specific_question(object_type)[0]

        input_ids = self.processor(text=question, add_special_tokens=False).input_ids
        input_ids = [self.processor.tokenizer.cls_token_id] + input_ids
        input_ids = torch.tensor(input_ids).unsqueeze(0)

        generated_ids = self.model.generate(pixel_values=pixel_values, input_ids=input_ids, max_length=50)
        caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return object_type + ": " + caption
        # return object_type

    def generate_embeddings(self, caption: str) -> np.ndarray:
        """Generate embeddings from caption text."""
        embeddings = self.embedding_model.encode(caption)
        return embeddings
    
    def process_uncaptioned_detections(self, batch_size: int = 10):
        """Process all uncaptioned detections in the database."""
        
        # Get all uncaptioned detections
        uncaptioned = self.db.get_uncaptioned_detections()
        
        if not uncaptioned:
            print("No uncaptioned detections found.")
            return
        
        print(f"Found {len(uncaptioned)} uncaptioned detections to process.")
        
        processed_count = 0
        
        for i in range(0, len(uncaptioned), batch_size):
            batch = uncaptioned[i:i + batch_size]
            
            for detection_id, object_type, crop_bytes, timestamp in batch:
                try:
                    # Convert bytes back to image
                    image = self.db.bytes_to_image(crop_bytes)
                    
                    # Convert OpenCV image (BGR) to PIL Image (RGB)
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(image_rgb)
                    
                    # Generate detailed caption with object-specific questioning
                    print(f"Processing {object_type} (ID: {detection_id})...")
                    question = self.get_object_specific_question(object_type)
                    print(f"Question used: {question}")
                    
                    caption = self.generate_detailed_caption(pil_image, object_type)
                    
                    # Generate embeddings
                    embeddings = self.generate_embeddings(caption)
                    
                    # Update database
                    self.db.update_caption_and_embeddings(
                        detection_id, caption, embeddings
                    )
                    
                    processed_count += 1
                    print(f"Processed {processed_count}/{len(uncaptioned)}: "
                          f"ID {detection_id} ({object_type})")
                    print(f"Generated Caption: {caption}")
                    print("-" * 80)
                    
                except Exception as e:
                    print(f"Error processing detection {detection_id}: {str(e)}")
                    continue
        
        print(f"\nCaptioning complete! Processed {processed_count} detections.")
    
    def process_single_detection(self, detection_id: int):
        """Process a single detection by ID."""
        
        detection = self.db.get_detection_by_id(detection_id)
        
        if not detection:
            print(f"Detection with ID {detection_id} not found.")
            return
        
        # Extract data from detection tuple - updated structure
        (det_id, object_type, timestamp, crop_bytes, original_video_link, 
         frame_num_original_video, caption, embeddings, confidence, bbox_x, bbox_y, 
         bbox_width, bbox_height, created_at) = detection
        
        if caption is not None:
            print(f"Detection {detection_id} already has a caption: {caption}")
            return
        
        try:
            # Convert bytes back to image
            image = self.db.bytes_to_image(crop_bytes)
            
            # Convert OpenCV image (BGR) to PIL Image (RGB)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            # Generate detailed caption
            caption = self.generate_detailed_caption(pil_image, object_type)
            
            # Generate embeddings
            embeddings = self.generate_embeddings(caption)
            
            # Update database
            self.db.update_caption_and_embeddings(detection_id, caption, embeddings)
            
            print(f"Successfully captioned detection {detection_id} ({object_type}):")
            print(f"Caption: {caption}")
            
        except Exception as e:
            print(f"Error processing detection {detection_id}: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description='Caption detected objects in database using Salesforce GIT TextVQA model')
    parser.add_argument('--db', default='detections.db',
                       help='Database path (default: detections.db)')
    parser.add_argument('--batch-size', type=int, default=10,
                       help='Batch size for processing (default: 10)')
    parser.add_argument('--detection-id', type=int,
                       help='Process single detection by ID')
    
    args = parser.parse_args()
    
    # Initialize caption generator
    captioner = CaptionGenerator(db_path=args.db)
    
    if args.detection_id:
        # Process single detection
        captioner.process_single_detection(args.detection_id)
    else:
        # Process all uncaptioned detections
        captioner.process_uncaptioned_detections(batch_size=args.batch_size)


if __name__ == "__main__":
    main()