"""
ReID (Re-Identification) feature extractor for person and vehicle clustering.
Supports TorchReID (deep-person-reid) for person/vehicle re-identification.
"""

import numpy as np
import cv2
import torch
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class ReIDExtractor:
    """Extract ReID embeddings for person and vehicle re-identification."""

    def __init__(self, model_type='person', use_gpu=True):
        """Initialize ReID extractor.

        Args:
            model_type: 'person' or 'vehicle'
            use_gpu: Whether to use GPU
        """
        self.model_type = model_type
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        self.model = None
        self.transform = None

    def init_model(self):
        """Initialize TorchReID model."""
        try:
            import torchreid

            logger.info(f"Initializing TorchReID {self.model_type} model...")

            if self.model_type == 'person':
                # Use pretrained person ReID model
                # Available models: osnet_x1_0, osnet_x0_75, osnet_x0_5, osnet_x0_25
                # resnet50, densenet121, etc.
                model_name = 'osnet_x1_0'
                logger.info(f"Loading {model_name} for person ReID...")

                self.model = torchreid.models.build_model(
                    name=model_name,
                    num_classes=1000,  # Will be ignored for feature extraction
                    loss='softmax',
                    pretrained=True
                )

            else:  # vehicle
                # Use ResNet50 for vehicle ReID (TorchReID focuses on person ReID)
                # For better vehicle ReID, we use a person model which generalizes well
                model_name = 'osnet_x1_0'
                logger.info(f"Loading {model_name} for vehicle ReID...")

                self.model = torchreid.models.build_model(
                    name=model_name,
                    num_classes=1000,
                    loss='softmax',
                    pretrained=True
                )

            # Move model to device and set to eval mode
            self.model = self.model.to(self.device)
            self.model.eval()

            # Setup transforms (standard ReID preprocessing)
            import torchvision.transforms as T
            self.transform = T.Compose([
                T.Resize((256, 128)),  # Standard ReID input size
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            logger.info(f"TorchReID {self.model_type} model ({model_name}) initialized on {self.device}")
            return True

        except ImportError:
            logger.error("TorchReID not installed. Install with: pip install torchreid")
            logger.info("Falling back to basic model configuration...")
            return self._init_fallback_model()

        except Exception as e:
            logger.error(f"Failed to initialize TorchReID model: {e}")
            logger.info("Falling back to basic model configuration...")
            return self._init_fallback_model()

    def _init_fallback_model(self):
        """Initialize fallback ResNet50 model if TorchReID is not available."""
        try:
            import torchvision.transforms as T
            from torchvision.models import resnet50

            logger.info("Loading fallback ResNet50 model...")
            self.model = resnet50(pretrained=True)
            # Remove classification head, use features
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
            self.model.to(self.device)
            self.model.eval()

            self.transform = T.Compose([
                T.Resize((256, 128)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            logger.info("Fallback ResNet50 model loaded successfully")
            return True

        except Exception as fallback_e:
            logger.error(f"Failed to load fallback model: {fallback_e}")
            return False

    def extract_features(self, image_bytes):
        """Extract ReID features from image bytes.

        Args:
            image_bytes: Image data as bytes

        Returns:
            Feature vector (numpy array) or None on error
        """
        if self.model is None:
            return None

        try:
            # Decode image
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                return None

            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)

            # Apply transforms
            img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

            # Extract features
            with torch.no_grad():
                features = self.model(img_tensor)

                # Flatten and normalize
                features = features.squeeze()
                if len(features.shape) > 1:
                    features = features.flatten()
                features = features / (features.norm() + 1e-7)

            return features.cpu().numpy()

        except Exception as e:
            logger.warning(f"Failed to extract ReID features: {e}")
            return None

    def extract_batch_features(self, image_bytes_list):
        """Extract ReID features from batch of images.

        Args:
            image_bytes_list: List of image data as bytes

        Returns:
            List of feature vectors
        """
        features_list = []
        for img_bytes in image_bytes_list:
            features = self.extract_features(img_bytes)
            features_list.append(features)
        return features_list
