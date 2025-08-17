from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import uuid
from datetime import datetime
import json
import pandas as pd
import numpy as np
import math
from io import StringIO
import csv


ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router
api_router = APIRouter()


# ID3 Decision Tree Implementation
class TreeNode:
    def __init__(self, feature=None, value=None, children=None, class_label=None, 
                 samples=None, entropy=None, info_gain=None):
        self.feature = feature  # Feature to split on
        self.value = value  # Value for leaf nodes
        self.children = children or {}  # Dictionary of feature_value -> TreeNode
        self.class_label = class_label  # For leaf nodes
        self.samples = samples or []  # Sample indices at this node
        self.entropy = entropy  # Entropy at this node
        self.info_gain = info_gain  # Information gain from this split
        self.id = str(uuid.uuid4())  # Unique ID for visualization

    def to_dict(self):
        """Convert tree node to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'feature': self.feature,
            'value': self.value,
            'class_label': self.class_label,
            'entropy': self.entropy,
            'info_gain': self.info_gain,
            'sample_count': len(self.samples),
            'children': {k: v.to_dict() for k, v in self.children.items()} if self.children else {}
        }


class DecisionTreeID3:
    def __init__(self):
        self.root = None
        self.steps = []  # Store step-by-step process for visualization
        self.feature_names = []
        self.target_name = ""
        
    def calculate_entropy(self, y):
        """Calculate entropy of target variable"""
        if len(y) == 0:
            return 0
        
        counts = {}
        for label in y:
            counts[label] = counts.get(label, 0) + 1
        
        entropy = 0
        total = len(y)
        for count in counts.values():
            if count > 0:
                prob = count / total
                entropy -= prob * math.log2(prob)
        
        return entropy
    
    def calculate_info_gain(self, X, y, feature_idx):
        """Calculate information gain for a feature"""
        if len(y) == 0:
            return 0
        
        # Calculate entropy before split
        entropy_before = self.calculate_entropy(y)
        
        # Get unique values for this feature
        feature_values = {}
        for i, value in enumerate(X[:, feature_idx]):
            if value not in feature_values:
                feature_values[value] = []
            feature_values[value].append(i)
        
        # Calculate weighted entropy after split
        entropy_after = 0
        total_samples = len(y)
        
        for value, indices in feature_values.items():
            subset_y = [y[i] for i in indices]
            subset_entropy = self.calculate_entropy(subset_y)
            weight = len(subset_y) / total_samples
            entropy_after += weight * subset_entropy
        
        return entropy_before - entropy_after
    
    def get_best_feature(self, X, y, available_features):
        """Find the feature with highest information gain"""
        if not available_features:
            return None
        
        best_feature = None
        best_gain = -1
        gains = {}
        
        for feature_idx in available_features:
            gain = self.calculate_info_gain(X, y, feature_idx)
            gains[feature_idx] = gain
            if gain > best_gain:
                best_gain = gain
                best_feature = feature_idx
        
        # Store step information
        step_info = {
            'step_type': 'feature_selection',
            'available_features': [self.feature_names[i] for i in available_features],
            'information_gains': {self.feature_names[i]: gains[i] for i in available_features},
            'selected_feature': self.feature_names[best_feature] if best_feature is not None else None,
            'best_gain': best_gain
        }
        self.steps.append(step_info)
        
        return best_feature
    
    def build_tree(self, X, y, available_features, samples, depth=0, max_depth=10):
        """Recursively build the decision tree"""
        # Create node
        node = TreeNode()
        node.samples = samples
        node.entropy = self.calculate_entropy([y[i] for i in samples])
        
        # Base cases
        subset_y = [y[i] for i in samples]
        
        # If all samples have same class, create leaf
        if len(set(subset_y)) == 1:
            node.class_label = subset_y[0]
            step_info = {
                'step_type': 'leaf_creation',
                'reason': 'pure_class',
                'class_label': node.class_label,
                'samples': len(samples)
            }
            self.steps.append(step_info)
            return node
        
        # If no more features or max depth reached, create leaf with majority class
        if not available_features or depth >= max_depth:
            # Get majority class
            class_counts = {}
            for label in subset_y:
                class_counts[label] = class_counts.get(label, 0) + 1
            node.class_label = max(class_counts.keys(), key=lambda x: class_counts[x])
            
            step_info = {
                'step_type': 'leaf_creation',
                'reason': 'no_features' if not available_features else 'max_depth',
                'class_label': node.class_label,
                'samples': len(samples)
            }
            self.steps.append(step_info)
            return node
        
        # Find best feature to split on
        best_feature = self.get_best_feature(
            X[samples], 
            [y[i] for i in samples], 
            available_features
        )
        
        if best_feature is None:
            # No good split found, create leaf
            class_counts = {}
            for label in subset_y:
                class_counts[label] = class_counts.get(label, 0) + 1
            node.class_label = max(class_counts.keys(), key=lambda x: class_counts[x])
            return node
        
        node.feature = self.feature_names[best_feature]
        node.info_gain = self.calculate_info_gain(X, y, best_feature)
        
        # Split data based on feature values
        feature_values = {}
        for i in samples:
            value = X[i, best_feature]
            if value not in feature_values:
                feature_values[value] = []
            feature_values[value].append(i)
        
        # Create children for each feature value
        remaining_features = [f for f in available_features if f != best_feature]
        
        for value, child_samples in feature_values.items():
            if child_samples:
                child_node = self.build_tree(
                    X, y, remaining_features, child_samples, depth + 1, max_depth
                )
                node.children[value] = child_node
        
        return node
    
    def fit(self, X, y, feature_names, target_name):
        """Train the decision tree"""
        self.feature_names = feature_names
        self.target_name = target_name
        self.steps = []
        
        # Convert to numpy array if not already
        if isinstance(X, list):
            X = np.array(X)
        
        available_features = list(range(X.shape[1]))
        samples = list(range(len(y)))
        
        # Add initial step
        initial_entropy = self.calculate_entropy(y)
        step_info = {
            'step_type': 'initialization',
            'total_samples': len(y),
            'features': feature_names,
            'target': target_name,
            'initial_entropy': initial_entropy,
            'class_distribution': {label: y.count(label) for label in set(y)}
        }
        self.steps.append(step_info)
        
        self.root = self.build_tree(X, y, available_features, samples)
        return self
    
    def predict_sample(self, x):
        """Predict class for a single sample"""
        node = self.root
        path = []
        
        while node.class_label is None:
            if node.feature is None:
                break
            
            feature_idx = self.feature_names.index(node.feature)
            feature_value = x[feature_idx]
            path.append((node.feature, feature_value))
            
            if feature_value in node.children:
                node = node.children[feature_value]
            else:
                # Handle unseen feature value - return majority class of current node
                break
        
        return node.class_label, path
    
    def get_tree_json(self):
        """Get tree structure as JSON for visualization"""
        if self.root:
            return self.root.to_dict()
        return None


# Pydantic Models
class DatasetRow(BaseModel):
    values: List[Any]

class ManualDataInput(BaseModel):
    feature_names: List[str]
    target_name: str
    data: List[DatasetRow]

class PredictionRequest(BaseModel):
    values: List[Any]

class TreeResponse(BaseModel):
    tree: Dict
    steps: List[Dict]
    feature_names: List[str]
    target_name: str

class DatasetInfo(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    feature_names: List[str]
    target_name: str
    data: List[List[Any]]
    created_at: datetime = Field(default_factory=datetime.utcnow)

# Sample datasets
SAMPLE_DATASETS = [
    {
        "name": "Weather Decision",
        "feature_names": ["Weather", "Temperature", "Humidity", "Wind"],
        "target_name": "Play",
        "data": [
            ["Sunny", "Hot", "High", "Weak", "No"],
            ["Sunny", "Hot", "High", "Strong", "No"],
            ["Overcast", "Hot", "High", "Weak", "Yes"],
            ["Rainy", "Mild", "High", "Weak", "Yes"],
            ["Rainy", "Cool", "Normal", "Weak", "Yes"],
            ["Rainy", "Cool", "Normal", "Strong", "No"],
            ["Overcast", "Cool", "Normal", "Strong", "Yes"],
            ["Sunny", "Mild", "High", "Weak", "No"],
            ["Sunny", "Cool", "Normal", "Weak", "Yes"],
            ["Rainy", "Mild", "Normal", "Weak", "Yes"],
            ["Sunny", "Mild", "Normal", "Strong", "Yes"],
            ["Overcast", "Mild", "High", "Strong", "Yes"],
            ["Overcast", "Hot", "Normal", "Weak", "Yes"],
            ["Rainy", "Mild", "High", "Strong", "No"]
        ]
    },
    {
        "name": "Animal Classification",
        "feature_names": ["Hair", "Feathers", "Eggs", "Milk"],
        "target_name": "Animal",
        "data": [
            ["Yes", "No", "No", "Yes", "Mammal"],
            ["No", "Yes", "Yes", "No", "Bird"],
            ["No", "No", "Yes", "No", "Reptile"],
            ["Yes", "No", "No", "Yes", "Mammal"],
            ["No", "Yes", "Yes", "No", "Bird"],
            ["No", "No", "Yes", "No", "Reptile"],
            ["Yes", "No", "Yes", "No", "Platypus"],
            ["No", "Yes", "Yes", "No", "Bird"],
        ]
    }
]

# Routes
@api_router.get("/")
async def root():
    return {"message": "Decision Tree ID3 Visualization API"}

@api_router.get("/sample-datasets")
async def get_sample_datasets():
    """Get available sample datasets"""
    return {"datasets": SAMPLE_DATASETS}

@api_router.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    """Upload and process CSV file"""
    try:
        contents = await file.read()
        csv_content = contents.decode('utf-8')
        
        # Parse CSV
        csv_reader = csv.reader(StringIO(csv_content))
        rows = list(csv_reader)
        
        if len(rows) < 2:
            raise HTTPException(status_code=400, detail="CSV must have at least header and one data row")
        
        # First row is header (features + target)
        headers = rows[0]
        feature_names = headers[:-1]
        target_name = headers[-1]
        
        # Remaining rows are data
        data = []
        for row in rows[1:]:
            if len(row) == len(headers):
                data.append(row)
        
        dataset = DatasetInfo(
            name=file.filename,
            feature_names=feature_names,
            target_name=target_name,
            data=data
        )
        
        # Store in database
        await db.datasets.insert_one(dataset.dict())
        
        return {
            "dataset_id": dataset.id,
            "feature_names": feature_names,
            "target_name": target_name,
            "rows": len(data),
            "data": data,  # Include full data for tree building
            "preview": data[:5]  # First 5 rows for preview
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing CSV: {str(e)}")

@api_router.post("/manual-data")
async def submit_manual_data(data_input: ManualDataInput):
    """Submit manually entered data"""
    try:
        # Convert data to list format
        data_rows = []
        for row in data_input.data:
            # Add target value (last value in the row)
            data_rows.append(row.values)
        
        dataset = DatasetInfo(
            name="Manual Entry",
            feature_names=data_input.feature_names,
            target_name=data_input.target_name,
            data=data_rows
        )
        
        # Store in database
        await db.datasets.insert_one(dataset.dict())
        
        return {
            "dataset_id": dataset.id,
            "feature_names": data_input.feature_names,
            "target_name": data_input.target_name,
            "rows": len(data_rows),
            "data": data_rows
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing manual data: {str(e)}")

@api_router.post("/build-tree")
async def build_decision_tree(dataset_data: Dict):
    """Build decision tree from dataset"""
    try:
        # Debug logging
        logger.info(f"Received dataset_data keys: {list(dataset_data.keys())}")
        logger.info(f"Dataset data type: {type(dataset_data)}")
        
        # Extract data with better error checking
        if "feature_names" not in dataset_data:
            raise ValueError("Missing 'feature_names' in dataset")
        if "target_name" not in dataset_data:
            raise ValueError("Missing 'target_name' in dataset")
        if "data" not in dataset_data:
            raise ValueError("Missing 'data' in dataset")
            
        feature_names = dataset_data["feature_names"]
        target_name = dataset_data["target_name"]
        data = dataset_data["data"]
        
        logger.info(f"Feature names: {feature_names}")
        logger.info(f"Target name: {target_name}")
        logger.info(f"Data length: {len(data)}")
        logger.info(f"First row: {data[0] if data else 'No data'}")
        
        # Validate data format
        if not data:
            raise ValueError("Dataset contains no data rows")
        
        # Check if data rows have the correct length
        expected_length = len(feature_names) + 1  # features + target
        for i, row in enumerate(data):
            if len(row) != expected_length:
                raise ValueError(f"Row {i} has {len(row)} columns, expected {expected_length}")
        
        # Separate features and target
        X = []
        y = []
        
        for row in data:
            X.append(row[:-1])  # All columns except last
            y.append(row[-1])   # Last column is target
        
        logger.info(f"X sample: {X[0] if X else 'No X data'}")
        logger.info(f"y sample: {y[0] if y else 'No y data'}")
        
        # Build tree
        tree_model = DecisionTreeID3()
        tree_model.fit(X, y, feature_names, target_name)
        
        # Get tree structure and steps
        tree_json = tree_model.get_tree_json()
        steps = tree_model.steps
        
        return TreeResponse(
            tree=tree_json,
            steps=steps,
            feature_names=feature_names,
            target_name=target_name
        )
        
    except Exception as e:
        logger.error(f"Error in build_decision_tree: {str(e)}")
        logger.error(f"Exception type: {type(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=f"Error building tree: {str(e)}")

@api_router.get("/datasets/{dataset_id}")
async def get_dataset(dataset_id: str):
    """Get dataset by ID"""
    dataset = await db.datasets.find_one({"id": dataset_id}, {"_id": 0})
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset

# Include the router in the main app
app.include_router(api_router, prefix="/api")

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"], # Allow all origins for development
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def shutdown_db_client():
    client.close()

app.add_event_handler("shutdown", shutdown_db_client)
