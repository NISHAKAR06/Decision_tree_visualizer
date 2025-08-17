#!/usr/bin/env python3
"""
Comprehensive Backend Testing for Decision Tree ID3 Visualization App
Tests all backend APIs including ID3 algorithm implementation
"""

import requests
import json
import math
import tempfile
import os
from io import StringIO

# Backend URL from environment
BACKEND_URL = "https://tree-viz-app.preview.emergentagent.com/api"

class BackendTester:
    def __init__(self):
        self.session = requests.Session()
        self.test_results = []
        
    def log_test(self, test_name, success, message="", details=None):
        """Log test results"""
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")
        if message:
            print(f"   {message}")
        if details:
            print(f"   Details: {details}")
        
        self.test_results.append({
            'test': test_name,
            'success': success,
            'message': message,
            'details': details
        })
        print()
    
    def test_root_endpoint(self):
        """Test GET /api/ endpoint"""
        try:
            response = self.session.get(f"{BACKEND_URL}/")
            
            if response.status_code == 200:
                data = response.json()
                if "message" in data:
                    self.log_test("Root Endpoint", True, f"Response: {data['message']}")
                else:
                    self.log_test("Root Endpoint", False, "Missing 'message' field in response")
            else:
                self.log_test("Root Endpoint", False, f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            self.log_test("Root Endpoint", False, f"Request failed: {str(e)}")
    
    def test_sample_datasets(self):
        """Test GET /api/sample-datasets endpoint"""
        try:
            response = self.session.get(f"{BACKEND_URL}/sample-datasets")
            
            if response.status_code == 200:
                data = response.json()
                
                if "datasets" not in data:
                    self.log_test("Sample Datasets", False, "Missing 'datasets' field in response")
                    return
                
                datasets = data["datasets"]
                
                # Check if we have expected datasets
                expected_datasets = ["Weather Decision", "Animal Classification"]
                found_datasets = [d["name"] for d in datasets]
                
                missing = [name for name in expected_datasets if name not in found_datasets]
                
                if missing:
                    self.log_test("Sample Datasets", False, f"Missing datasets: {missing}")
                else:
                    # Validate dataset structure
                    valid = True
                    for dataset in datasets:
                        required_fields = ["name", "feature_names", "target_name", "data"]
                        for field in required_fields:
                            if field not in dataset:
                                valid = False
                                self.log_test("Sample Datasets", False, f"Dataset '{dataset.get('name', 'unknown')}' missing field: {field}")
                                break
                    
                    if valid:
                        self.log_test("Sample Datasets", True, f"Found {len(datasets)} valid datasets: {found_datasets}")
                        
                        # Test Weather Decision dataset structure
                        weather_dataset = next((d for d in datasets if d["name"] == "Weather Decision"), None)
                        if weather_dataset:
                            expected_features = ["Weather", "Temperature", "Humidity", "Wind"]
                            if weather_dataset["feature_names"] == expected_features and weather_dataset["target_name"] == "Play":
                                self.log_test("Weather Dataset Structure", True, "Correct features and target")
                            else:
                                self.log_test("Weather Dataset Structure", False, f"Expected features: {expected_features}, target: 'Play'")
            else:
                self.log_test("Sample Datasets", False, f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            self.log_test("Sample Datasets", False, f"Request failed: {str(e)}")
    
    def test_csv_upload(self):
        """Test POST /api/upload-csv endpoint"""
        try:
            # Create a test CSV file
            csv_content = """Weather,Temperature,Humidity,Play
Sunny,Hot,High,No
Rainy,Cool,Normal,Yes
Overcast,Mild,High,Yes
Sunny,Cool,Normal,Yes
Rainy,Hot,High,No"""
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                f.write(csv_content)
                temp_file_path = f.name
            
            try:
                # Upload the CSV file
                with open(temp_file_path, 'rb') as f:
                    files = {'file': ('test_data.csv', f, 'text/csv')}
                    response = self.session.post(f"{BACKEND_URL}/upload-csv", files=files)
                
                if response.status_code == 200:
                    data = response.json()
                    required_fields = ["dataset_id", "feature_names", "target_name", "rows"]
                    
                    missing_fields = [field for field in required_fields if field not in data]
                    if missing_fields:
                        self.log_test("CSV Upload", False, f"Missing fields in response: {missing_fields}")
                    else:
                        expected_features = ["Weather", "Temperature", "Humidity"]
                        if (data["feature_names"] == expected_features and 
                            data["target_name"] == "Play" and 
                            data["rows"] == 5):
                            self.log_test("CSV Upload", True, f"Successfully uploaded CSV with {data['rows']} rows")
                            return data["dataset_id"]  # Return dataset ID for further testing
                        else:
                            self.log_test("CSV Upload", False, f"Incorrect parsing: features={data['feature_names']}, target={data['target_name']}, rows={data['rows']}")
                else:
                    self.log_test("CSV Upload", False, f"HTTP {response.status_code}: {response.text}")
                    
            finally:
                # Clean up temporary file
                os.unlink(temp_file_path)
                
        except Exception as e:
            self.log_test("CSV Upload", False, f"Request failed: {str(e)}")
        
        return None
    
    def test_manual_data_entry(self):
        """Test POST /api/manual-data endpoint"""
        try:
            # Test data for manual entry
            manual_data = {
                "feature_names": ["Weather", "Temperature"],
                "target_name": "Play",
                "data": [
                    {"values": ["Sunny", "Hot", "No"]},
                    {"values": ["Rainy", "Cool", "Yes"]},
                    {"values": ["Overcast", "Mild", "Yes"]},
                    {"values": ["Sunny", "Cool", "Yes"]},
                    {"values": ["Rainy", "Hot", "No"]}
                ]
            }
            
            response = self.session.post(
                f"{BACKEND_URL}/manual-data",
                json=manual_data,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                required_fields = ["dataset_id", "feature_names", "target_name", "rows"]
                
                missing_fields = [field for field in required_fields if field not in data]
                if missing_fields:
                    self.log_test("Manual Data Entry", False, f"Missing fields in response: {missing_fields}")
                else:
                    if (data["feature_names"] == manual_data["feature_names"] and 
                        data["target_name"] == manual_data["target_name"] and 
                        data["rows"] == len(manual_data["data"])):
                        self.log_test("Manual Data Entry", True, f"Successfully processed {data['rows']} manual data rows")
                        return data["dataset_id"]  # Return dataset ID for further testing
                    else:
                        self.log_test("Manual Data Entry", False, f"Data mismatch in response")
            else:
                self.log_test("Manual Data Entry", False, f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            self.log_test("Manual Data Entry", False, f"Request failed: {str(e)}")
        
        return None
    
    def test_tree_building(self):
        """Test POST /api/build-tree endpoint with ID3 algorithm validation"""
        try:
            # Test data for tree building
            tree_data = {
                "feature_names": ["Weather", "Temperature"],
                "target_name": "Play",
                "data": [
                    ["Sunny", "Hot", "No"],
                    ["Rainy", "Cool", "Yes"],
                    ["Overcast", "Mild", "Yes"],
                    ["Sunny", "Cool", "Yes"],
                    ["Rainy", "Hot", "No"]
                ]
            }
            
            response = self.session.post(
                f"{BACKEND_URL}/build-tree",
                json=tree_data,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                required_fields = ["tree", "steps", "feature_names", "target_name"]
                
                missing_fields = [field for field in required_fields if field not in data]
                if missing_fields:
                    self.log_test("Tree Building", False, f"Missing fields in response: {missing_fields}")
                    return
                
                # Validate tree structure
                tree = data["tree"]
                steps = data["steps"]
                
                tree_valid = self.validate_tree_structure(tree)
                steps_valid = self.validate_algorithm_steps(steps)
                math_valid = self.validate_mathematical_correctness(tree_data, tree, steps)
                
                if tree_valid and steps_valid and math_valid:
                    self.log_test("Tree Building", True, f"Successfully built tree with {len(steps)} algorithm steps")
                    self.log_test("ID3 Algorithm Implementation", True, "Mathematical calculations and tree structure are correct")
                else:
                    self.log_test("Tree Building", False, "Tree structure, steps, or mathematical calculations are invalid")
                    
            else:
                self.log_test("Tree Building", False, f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            self.log_test("Tree Building", False, f"Request failed: {str(e)}")
    
    def validate_tree_structure(self, tree):
        """Validate the tree structure returned by the API"""
        try:
            required_fields = ["id", "entropy", "sample_count"]
            
            def check_node(node):
                for field in required_fields:
                    if field not in node:
                        return False
                
                # Check if it's a leaf or internal node
                if "class_label" in node and node["class_label"] is not None:
                    # Leaf node
                    return True
                elif "feature" in node and node["feature"] is not None:
                    # Internal node - should have children
                    if "children" not in node:
                        return False
                    
                    # Recursively check children
                    for child in node["children"].values():
                        if not check_node(child):
                            return False
                    return True
                else:
                    return False
            
            return check_node(tree)
            
        except Exception as e:
            print(f"Tree validation error: {e}")
            return False
    
    def validate_algorithm_steps(self, steps):
        """Validate the algorithm steps returned by the API"""
        try:
            if not steps:
                return False
            
            # Check for required step types
            step_types = [step.get("step_type") for step in steps]
            
            # Should have initialization step
            if "initialization" not in step_types:
                return False
            
            # Should have at least one feature selection or leaf creation
            if not any(step_type in ["feature_selection", "leaf_creation"] for step_type in step_types):
                return False
            
            # Validate initialization step
            init_step = next((step for step in steps if step.get("step_type") == "initialization"), None)
            if init_step:
                required_init_fields = ["total_samples", "features", "target", "initial_entropy"]
                for field in required_init_fields:
                    if field not in init_step:
                        return False
            
            return True
            
        except Exception as e:
            print(f"Steps validation error: {e}")
            return False
    
    def validate_mathematical_correctness(self, input_data, tree, steps):
        """Validate mathematical correctness of ID3 calculations"""
        try:
            # Extract target values for entropy calculation
            target_values = [row[-1] for row in input_data["data"]]
            
            # Calculate expected initial entropy
            def calculate_entropy(labels):
                if not labels:
                    return 0
                
                counts = {}
                for label in labels:
                    counts[label] = counts.get(label, 0) + 1
                
                entropy = 0
                total = len(labels)
                for count in counts.values():
                    if count > 0:
                        prob = count / total
                        entropy -= prob * math.log2(prob)
                
                return entropy
            
            expected_entropy = calculate_entropy(target_values)
            
            # Find initialization step
            init_step = next((step for step in steps if step.get("step_type") == "initialization"), None)
            if init_step and "initial_entropy" in init_step:
                actual_entropy = init_step["initial_entropy"]
                
                # Allow small floating point differences
                if abs(expected_entropy - actual_entropy) < 0.0001:
                    return True
                else:
                    print(f"Entropy mismatch: expected {expected_entropy}, got {actual_entropy}")
                    return False
            
            return False
            
        except Exception as e:
            print(f"Mathematical validation error: {e}")
            return False
    
    def test_dataset_retrieval(self, dataset_id):
        """Test GET /api/datasets/{dataset_id} endpoint"""
        if not dataset_id:
            self.log_test("Dataset Retrieval", False, "No dataset ID provided for testing")
            return
        
        try:
            response = self.session.get(f"{BACKEND_URL}/datasets/{dataset_id}")
            
            if response.status_code == 200:
                data = response.json()
                required_fields = ["id", "name", "feature_names", "target_name", "data"]
                
                missing_fields = [field for field in required_fields if field not in data]
                if missing_fields:
                    self.log_test("Dataset Retrieval", False, f"Missing fields in response: {missing_fields}")
                else:
                    if data["id"] == dataset_id:
                        self.log_test("Dataset Retrieval", True, f"Successfully retrieved dataset: {data['name']}")
                    else:
                        self.log_test("Dataset Retrieval", False, f"Dataset ID mismatch: expected {dataset_id}, got {data['id']}")
            elif response.status_code == 404:
                self.log_test("Dataset Retrieval", False, f"Dataset not found: {dataset_id}")
            else:
                self.log_test("Dataset Retrieval", False, f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            self.log_test("Dataset Retrieval", False, f"Request failed: {str(e)}")
    
    def test_sample_dataset_tree_building(self):
        """Test tree building with sample Weather Decision dataset"""
        try:
            # First get sample datasets
            response = self.session.get(f"{BACKEND_URL}/sample-datasets")
            if response.status_code != 200:
                self.log_test("Sample Dataset Tree Building", False, "Could not fetch sample datasets")
                return
            
            datasets = response.json()["datasets"]
            weather_dataset = next((d for d in datasets if d["name"] == "Weather Decision"), None)
            
            if not weather_dataset:
                self.log_test("Sample Dataset Tree Building", False, "Weather Decision dataset not found")
                return
            
            # Build tree with weather dataset
            tree_response = self.session.post(
                f"{BACKEND_URL}/build-tree",
                json=weather_dataset,
                headers={"Content-Type": "application/json"}
            )
            
            if tree_response.status_code == 200:
                tree_data = tree_response.json()
                
                # Validate that tree was built successfully
                if "tree" in tree_data and "steps" in tree_data:
                    tree = tree_data["tree"]
                    steps = tree_data["steps"]
                    
                    # Check that we have reasonable number of steps for this dataset
                    if len(steps) >= 3:  # At least initialization + some feature selection/leaf creation
                        self.log_test("Sample Dataset Tree Building", True, f"Successfully built tree from Weather Decision dataset with {len(steps)} steps")
                    else:
                        self.log_test("Sample Dataset Tree Building", False, f"Too few algorithm steps: {len(steps)}")
                else:
                    self.log_test("Sample Dataset Tree Building", False, "Missing tree or steps in response")
            else:
                self.log_test("Sample Dataset Tree Building", False, f"HTTP {tree_response.status_code}: {tree_response.text}")
                
        except Exception as e:
            self.log_test("Sample Dataset Tree Building", False, f"Request failed: {str(e)}")
    
    def run_all_tests(self):
        """Run all backend tests"""
        print("=" * 60)
        print("DECISION TREE ID3 BACKEND TESTING")
        print("=" * 60)
        print()
        
        # Test basic endpoints
        self.test_root_endpoint()
        self.test_sample_datasets()
        
        # Test data input methods
        csv_dataset_id = self.test_csv_upload()
        manual_dataset_id = self.test_manual_data_entry()
        
        # Test tree building and ID3 algorithm
        self.test_tree_building()
        self.test_sample_dataset_tree_building()
        
        # Test dataset retrieval
        if csv_dataset_id:
            self.test_dataset_retrieval(csv_dataset_id)
        if manual_dataset_id:
            self.test_dataset_retrieval(manual_dataset_id)
        
        # Summary
        print("=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        passed = sum(1 for result in self.test_results if result['success'])
        total = len(self.test_results)
        
        print(f"Total Tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {total - passed}")
        print(f"Success Rate: {(passed/total)*100:.1f}%")
        
        if total - passed > 0:
            print("\nFailed Tests:")
            for result in self.test_results:
                if not result['success']:
                    print(f"  - {result['test']}: {result['message']}")
        
        return passed == total

if __name__ == "__main__":
    tester = BackendTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n🎉 All backend tests passed!")
    else:
        print("\n⚠️  Some backend tests failed. Check the details above.")