import React, { useState, useEffect } from "react";
import "./App.css";
import axios from "axios";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

// Tree Visualization Component
const TreeVisualization = ({ treeData, onNodeClick }) => {
  const renderNode = (node, x = 400, y = 100, level = 0) => {
    if (!node) return null;

    const nodeRadius = 40;
    const levelHeight = 120;
    const childrenCount = Object.keys(node.children).length;
    const childWidth = 200;

    return (
      <g key={node.id}>
        {/* Node circle */}
        <circle
          cx={x}
          cy={y}
          r={nodeRadius}
          fill={node.class_label ? "#4ade80" : "#3b82f6"}
          stroke="#1e40af"
          strokeWidth="2"
          className="cursor-pointer hover:opacity-80"
          onClick={() => onNodeClick(node)}
        />
        
        {/* Node label */}
        <text
          x={x}
          y={y - 5}
          textAnchor="middle"
          className="text-sm font-semibold fill-white pointer-events-none"
        >
          {node.feature || node.class_label}
        </text>
        
        {/* Node info */}
        <text
          x={x}
          y={y + 8}
          textAnchor="middle"
          className="text-xs fill-white pointer-events-none"
        >
          {node.sample_count} samples
        </text>

        {/* Children */}
        {Object.entries(node.children).map(([value, child], index) => {
          const childX = x + (index - (childrenCount - 1) / 2) * childWidth;
          const childY = y + levelHeight;

          return (
            <g key={`${node.id}-${value}`}>
              {/* Edge */}
              <line
                x1={x}
                y1={y + nodeRadius}
                x2={childX}
                y2={childY - nodeRadius}
                stroke="#6b7280"
                strokeWidth="2"
              />
              
              {/* Edge label */}
              <text
                x={(x + childX) / 2}
                y={(y + childY) / 2 - 10}
                textAnchor="middle"
                className="text-xs fill-gray-600 font-medium"
              >
                {value}
              </text>

              {/* Recursive child rendering */}
              {renderNode(child, childX, childY, level + 1)}
            </g>
          );
        })}
      </g>
    );
  };

  if (!treeData) {
    return (
      <div className="flex items-center justify-center h-64 bg-gray-100 rounded-lg">
        <p className="text-gray-500">No tree data available</p>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-lg p-4 h-[600px] overflow-auto">
      <h3 className="text-lg font-semibold mb-4">Decision Tree Visualization</h3>
      <svg width="800" height="800" className="border border-gray-200 rounded">
        {renderNode(treeData)}
      </svg>
    </div>
  );
};

// Algorithm Steps Component
const AlgorithmSteps = ({ steps }) => {
  const [currentStep, setCurrentStep] = useState(0);

  useEffect(() => {
    setCurrentStep(0);
  }, [steps]);

  if (!steps || steps.length === 0) {
    return (
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h3 className="text-lg font-semibold mb-4">Algorithm Steps</h3>
        <p className="text-gray-500">No steps available</p>
      </div>
    );
  }

  const step = steps[currentStep];

  if (!step) {
    return (
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h3 className="text-lg font-semibold mb-4">Algorithm Steps</h3>
        <p className="text-gray-500">Error: Step not found.</p>
      </div>
    );
  }

  const renderStepContent = () => {
    switch (step.step_type) {
      case 'initialization':
        return (
          <div>
            <h4 className="font-semibold text-blue-600">Initialization</h4>
            <p>Total samples: {step.total_samples}</p>
            <p>Features: {step.features.join(', ')}</p>
            <p>Target: {step.target}</p>
            <p>Initial entropy: {step.initial_entropy.toFixed(4)}</p>
            <div className="mt-2">
              <h5 className="font-medium">Class distribution:</h5>
              {Object.entries(step.class_distribution).map(([label, count]) => (
                <span key={label} className="inline-block bg-blue-100 px-2 py-1 rounded mr-2 mt-1">
                  {label}: {count}
                </span>
              ))}
            </div>
          </div>
        );
      
      case 'feature_selection':
        return (
          <div>
            <h4 className="font-semibold text-green-600">Feature Selection</h4>
            <p>Available features: {step.available_features.join(', ')}</p>
            <div className="mt-2">
              <h5 className="font-medium">Information gains:</h5>
              {Object.entries(step.information_gains).map(([feature, gain]) => (
                <div key={feature} className={`p-2 rounded mt-1 ${feature === step.selected_feature ? 'bg-green-100' : 'bg-gray-100'}`}>
                  {feature}: {gain.toFixed(4)}
                </div>
              ))}
            </div>
            <p className="mt-2 font-medium text-green-700">
              Selected: {step.selected_feature} (gain: {step.best_gain.toFixed(4)})
            </p>
          </div>
        );
      
      case 'leaf_creation':
        return (
          <div>
            <h4 className="font-semibold text-red-600">Leaf Node Created</h4>
            <p>Reason: {step.reason}</p>
            <p>Class label: {step.class_label}</p>
            <p>Samples: {step.samples}</p>
          </div>
        );
      
      default:
        return <div>Unknown step type</div>;
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold">Algorithm Steps</h3>
        <span className="text-sm text-gray-500">
          Step {currentStep + 1} of {steps.length}
        </span>
      </div>
      
      <div className="mb-4 p-4 border rounded-lg bg-gray-50">
        {renderStepContent()}
      </div>
      
      <div className="flex justify-between">
        <button
          onClick={() => setCurrentStep(Math.max(0, currentStep - 1))}
          disabled={currentStep === 0}
          className="px-4 py-2 bg-blue-500 text-white rounded disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Previous
        </button>
        
        <button
          onClick={() => setCurrentStep(Math.min(steps.length - 1, currentStep + 1))}
          disabled={currentStep === steps.length - 1}
          className="px-4 py-2 bg-blue-500 text-white rounded disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Next
        </button>
      </div>
    </div>
  );
};

// Data Input Components
const CSVUpload = ({ onDataLoad }) => {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);

  const handleFileUpload = async () => {
    if (!file) return;

    setUploading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(`${API}/upload-csv`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      onDataLoad(response.data);
    } catch (error) {
      alert('Error uploading file: ' + error.response?.data?.detail);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="bg-white p-6 rounded-lg shadow">
      <h3 className="text-lg font-semibold mb-4">Upload CSV File</h3>
      <div className="space-y-4">
        <input
          type="file"
          accept=".csv"
          onChange={(e) => setFile(e.target.files[0])}
          className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
        />
        <button
          onClick={handleFileUpload}
          disabled={!file || uploading}
          className="px-4 py-2 bg-blue-500 text-white rounded disabled:opacity-50"
        >
          {uploading ? 'Uploading...' : 'Upload & Process'}
        </button>
      </div>
    </div>
  );
};

const ManualDataEntry = ({ onDataLoad }) => {
  const [features, setFeatures] = useState(['']);
  const [target, setTarget] = useState('');
  const [rows, setRows] = useState([[]]);

  const addFeature = () => setFeatures([...features, '']);
  const removeFeature = (index) => {
    const newFeatures = features.filter((_, i) => i !== index);
    setFeatures(newFeatures);
    // Update rows to match new feature count
    setRows(rows.map(row => row.filter((_, i) => i !== index)));
  };

  const addRow = () => setRows([...rows, new Array(features.length + 1).fill('')]);
  const removeRow = (index) => setRows(rows.filter((_, i) => i !== index));

  const updateFeature = (index, value) => {
    const newFeatures = [...features];
    newFeatures[index] = value;
    setFeatures(newFeatures);
  };

  const updateCell = (rowIndex, colIndex, value) => {
    const newRows = [...rows];
    if (!newRows[rowIndex]) newRows[rowIndex] = [];
    newRows[rowIndex][colIndex] = value;
    setRows(newRows);
  };

  const submitData = async () => {
    if (features.some(f => !f) || !target) {
      alert('Please fill all feature names and target name');
      return;
    }

    const data = rows
      .filter(row => row.some(cell => cell))
      .map(row => ({ values: [...row.slice(0, features.length), row[features.length] || ''] }));

    try {
      const response = await axios.post(`${API}/manual-data`, {
        feature_names: features.filter(f => f),
        target_name: target,
        data
      });
      onDataLoad(response.data);
    } catch (error) {
      alert('Error submitting data: ' + error.response?.data?.detail);
    }
  };

  return (
    <div className="bg-white p-6 rounded-lg shadow">
      <h3 className="text-lg font-semibold mb-4">Manual Data Entry</h3>
      
      {/* Feature names */}
      <div className="mb-4">
        <label className="block text-sm font-medium mb-2">Feature Names</label>
        {features.map((feature, index) => (
          <div key={index} className="flex mb-2">
            <input
              type="text"
              value={feature}
              onChange={(e) => updateFeature(index, e.target.value)}
              placeholder={`Feature ${index + 1}`}
              className="flex-1 px-3 py-2 border border-gray-300 rounded mr-2"
            />
            <button
              onClick={() => removeFeature(index)}
              className="px-3 py-2 bg-red-500 text-white rounded"
            >
              Remove
            </button>
          </div>
        ))}
        <button
          onClick={addFeature}
          className="px-4 py-2 bg-green-500 text-white rounded"
        >
          Add Feature
        </button>
      </div>

      {/* Target name */}
      <div className="mb-4">
        <label className="block text-sm font-medium mb-2">Target Variable</label>
        <input
          type="text"
          value={target}
          onChange={(e) => setTarget(e.target.value)}
          placeholder="Target variable name"
          className="w-full px-3 py-2 border border-gray-300 rounded"
        />
      </div>

      {/* Data rows */}
      <div className="mb-4">
        <label className="block text-sm font-medium mb-2">Data Rows</label>
        <div className="overflow-x-auto">
          <table className="min-w-full table-auto">
            <thead>
              <tr>
                {features.map((feature, index) => (
                  <th key={index} className="px-2 py-1 border text-left text-sm">
                    {feature || `Feature ${index + 1}`}
                  </th>
                ))}
                <th className="px-2 py-1 border text-left text-sm">{target || 'Target'}</th>
                <th className="px-2 py-1 border">Action</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row, rowIndex) => (
                <tr key={rowIndex}>
                  {features.map((_, colIndex) => (
                    <td key={colIndex} className="px-2 py-1 border">
                      <input
                        type="text"
                        value={row[colIndex] || ''}
                        onChange={(e) => updateCell(rowIndex, colIndex, e.target.value)}
                        className="w-full px-2 py-1 text-sm border-0 focus:outline-none"
                      />
                    </td>
                  ))}
                  <td className="px-2 py-1 border">
                    <input
                      type="text"
                      value={row[features.length] || ''}
                      onChange={(e) => updateCell(rowIndex, features.length, e.target.value)}
                      className="w-full px-2 py-1 text-sm border-0 focus:outline-none"
                    />
                  </td>
                  <td className="px-2 py-1 border text-center">
                    <button
                      onClick={() => removeRow(rowIndex)}
                      className="px-2 py-1 bg-red-500 text-white text-sm rounded"
                    >
                      Remove
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <button
          onClick={addRow}
          className="mt-2 px-4 py-2 bg-green-500 text-white rounded"
        >
          Add Row
        </button>
      </div>

      <button
        onClick={submitData}
        className="px-6 py-2 bg-blue-500 text-white rounded"
      >
        Submit Data
      </button>
    </div>
  );
};

const SampleDatasets = ({ onDataLoad }) => {
  const [datasets, setDatasets] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchSampleDatasets();
  }, []);

  const fetchSampleDatasets = async () => {
    try {
      const response = await axios.get(`${API}/sample-datasets`);
      setDatasets(response.data.datasets);
    } catch (error) {
      console.error('Error fetching sample datasets:', error);
    }
  };

  const selectDataset = (dataset) => {
    onDataLoad({
      feature_names: dataset.feature_names,
      target_name: dataset.target_name,
      data: dataset.data,
      rows: dataset.data.length
    });
  };

  return (
    <div className="bg-white p-6 rounded-lg shadow">
      <h3 className="text-lg font-semibold mb-4">Sample Datasets</h3>
      <div className="space-y-4">
        {datasets.map((dataset, index) => (
          <div key={index} className="border border-gray-200 rounded-lg p-4">
            <h4 className="font-medium mb-2">{dataset.name}</h4>
            <p className="text-sm text-gray-600 mb-2">
              Features: {dataset.feature_names.join(', ')}
            </p>
            <p className="text-sm text-gray-600 mb-3">
              Target: {dataset.target_name} ({dataset.data.length} samples)
            </p>
            <button
              onClick={() => selectDataset(dataset)}
              className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
            >
              Use This Dataset
            </button>
          </div>
        ))}
      </div>
    </div>
  );
};

// Main App Component
function App() {
  const [currentTab, setCurrentTab] = useState('upload');
  const [dataset, setDataset] = useState(null);
  const [treeData, setTreeData] = useState(null);
  const [algorithmSteps, setAlgorithmSteps] = useState([]);
  const [selectedNode, setSelectedNode] = useState(null);
  const [building, setBuilding] = useState(false);

  const handleDataLoad = (data) => {
    const newDataset = { ...data };
    if (!newDataset.data) {
      newDataset.data = [];
    }
    setDataset(newDataset);
    setTreeData(null);
    setAlgorithmSteps([]);
    setSelectedNode(null);
  };

  const buildTree = async () => {
    if (!dataset) return;

    setBuilding(true);
    try {
      const response = await axios.post(`${API}/build-tree`, dataset);
      setTreeData(response.data.tree);
      setAlgorithmSteps(response.data.steps);
    } catch (error) {
      alert('Error building tree: ' + error.response?.data?.detail);
    } finally {
      setBuilding(false);
    }
  };

  const handleNodeClick = (node) => {
    setSelectedNode(node);
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <h1 className="text-3xl font-bold text-gray-900">
            Decision Tree ID3 Visualization
          </h1>
          <p className="text-gray-600 mt-2">
            Visualize the ID3 algorithm step-by-step with interactive decision trees
          </p>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Left Column - Data Input */}
          <div className="space-y-6">
            {/* Tab Navigation */}
            <div className="bg-white rounded-lg shadow">
              <nav className="flex space-x-8 px-6 py-4 border-b">
                {[
                  { key: 'upload', label: 'CSV Upload' },
                  { key: 'manual', label: 'Manual Entry' },
                  { key: 'sample', label: 'Sample Data' }
                ].map((tab) => (
                  <button
                    key={tab.key}
                    onClick={() => setCurrentTab(tab.key)}
                    className={`py-2 px-1 border-b-2 font-medium text-sm ${
                      currentTab === tab.key
                        ? 'border-blue-500 text-blue-600'
                        : 'border-transparent text-gray-500 hover:text-gray-700'
                    }`}
                  >
                    {tab.label}
                  </button>
                ))}
              </nav>

              <div className="p-6">
                {currentTab === 'upload' && <CSVUpload onDataLoad={handleDataLoad} />}
                {currentTab === 'manual' && <ManualDataEntry onDataLoad={handleDataLoad} />}
                {currentTab === 'sample' && <SampleDatasets onDataLoad={handleDataLoad} />}
              </div>
            </div>

            {/* Dataset Info */}
            {dataset && (
              <div className="bg-white rounded-lg shadow p-6">
                <h3 className="text-lg font-semibold mb-4">Dataset Information</h3>
                <div className="space-y-2">
                  <p><span className="font-medium">Features:</span> {dataset.feature_names?.join(', ')}</p>
                  <p><span className="font-medium">Target:</span> {dataset.target_name}</p>
                  <p><span className="font-medium">Samples:</span> {dataset.rows}</p>
                </div>
                <button
                  onClick={buildTree}
                  disabled={building}
                  className="mt-4 px-6 py-2 bg-green-500 text-white rounded disabled:opacity-50"
                >
                  {building ? 'Building Tree...' : 'Build Decision Tree'}
                </button>
              </div>
            )}

            {/* Selected Node Info */}
            {selectedNode && (
              <div className="bg-white rounded-lg shadow p-6">
                <h3 className="text-lg font-semibold mb-4">Node Information</h3>
                <div className="space-y-2">
                  <p><span className="font-medium">Feature:</span> {selectedNode.feature || 'Leaf Node'}</p>
                  {selectedNode.class_label && (
                    <p><span className="font-medium">Class:</span> {selectedNode.class_label}</p>
                  )}
                  <p><span className="font-medium">Samples:</span> {selectedNode.sample_count}</p>
                  {selectedNode.entropy !== null && (
                    <p><span className="font-medium">Entropy:</span> {selectedNode.entropy?.toFixed(4)}</p>
                  )}
                  {selectedNode.info_gain !== null && (
                    <p><span className="font-medium">Information Gain:</span> {selectedNode.info_gain?.toFixed(4)}</p>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Right Column - Visualization */}
          <div className="space-y-6">
            <TreeVisualization treeData={treeData} onNodeClick={handleNodeClick} />
            <AlgorithmSteps steps={algorithmSteps} />
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
