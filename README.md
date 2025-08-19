# Decision Tree Visualizer

A web application for visualizing the ID3 decision tree algorithm, built with FastAPI (Python backend) and React (JavaScript frontend). This tool lets users upload datasets, view step-by-step tree construction, and interactively explore decision trees.

## Features

- **Upload datasets** (CSV format) for analysis.
- **Interactive decision tree visualization** using a custom React component.
- **Step-by-step algorithm breakdown** for educational purposes.
- **RESTful API backend** powered by FastAPI.
- **MongoDB integration** for storing data and results.
- **Modern UI** styled with Tailwind CSS.

## Project Structure

```
.
├── backend/
│   ├── server.py          # FastAPI app, decision tree logic (ID3), MongoDB connection
│   ├── .env               # Environment variables (MONGO_URL, DB_NAME, etc.)
│   └── requirements.txt   # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── App.js         # Main React app, visualization logic
│   │   ├── App.css        # Custom styles
│   │   └── index.js       # App entry point
│   ├── public/
│   │   └── index.html     # HTML template
│   ├── package.json       # Frontend dependencies
│   └── tailwind.config.js # Tailwind CSS config
├── .gitconfig             # Git configuration
```

> **Note:** This file listing is based on a limited search result, and may be incomplete. [View all files in GitHub](https://github.com/NISHAKAR06/Decision_tree_visualizer/search)

## Getting Started

### Backend

1. **Install dependencies:**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```
2. **Configure environment variables:**
   - Set MongoDB URL and DB name in `.env`.
3. **Run FastAPI server:**
   ```bash
   uvicorn server:app --reload
   ```

### Frontend

1. **Install dependencies:**
   ```bash
   cd frontend
   yarn install
   ```
2. **Start the React app:**
   ```bash
   yarn start
   ```

### Configuration

- **MongoDB:** Used for storing uploaded datasets and trees.
- **CORS:** Configured via backend `.env` file.
- **API Endpoint:** The frontend expects a `REACT_APP_BACKEND_URL` environment variable.

## Technologies Used

- **Backend:** FastAPI, Python, MongoDB, Motor, Pandas, Numpy
- **Frontend:** React, Axios, Tailwind CSS
- **Others:** Docker/Cloud support via requirements, ESLint, Prettier

## Example Usage

1. Upload a dataset (CSV) in the web UI.
2. View the constructed decision tree.
3. Explore each split with entropy and information gain.
4. Interact with tree nodes for more details.

## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.
