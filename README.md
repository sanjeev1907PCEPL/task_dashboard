# Task Management Dashboard

## Overview
The **Task Management Dashboard** is a **Streamlit-based web application** designed to visualize and manage task schedules for project orders. It calculates ideal start and completion dates based on task dependencies, lead times, and fallback sources, with an intuitive Gantt chart and status classification (e.g., "Completed On Time", "Overdue").  
Currently at **Version 2.6.0**, this project helps project managers and teams track progress efficiently.

- **Version**: 2.6.0
- **Date**: April 27, 2025
- **Author**: Developed with assistance from Grok 3 (xAI)

## Features
- Calculate ideal schedules with dependency management.
- Dynamically set dispatch due dates and lead times.
- Visualize tasks in a Gantt chart with status-based coloring.
- Classify tasks as "Completed On Time", "Completed Late", "Not Due Yet", "Overdue", or "No Data".
- Export schedules to Excel.
- Filter by order number and assignee.
- Detect and warn about dependency cycles.

## Prerequisites
- Python 3.8 or higher
- Git (for version control)
- Internet connection (for deployment)

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/task-management-dashboard.git
   cd task-management-dashboard
   ```

2. **Install Dependencies**  
   Ensure you have a `requirements.txt` file with the following dependencies:
   ```
   streamlit
   pandas
   plotly
   networkx
   xlsxwriter
   ```
   Install them using:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application Locally**
   Start the Streamlit app with:
   ```bash
   streamlit run task_dashboard.py
   ```
   Then, open your browser and go to `http://localhost:8501` to view the dashboard.

## Usage
- **Upload Data**: Upload a Microsoft Planner Excel file containing columns such as Task No, Activity Name, Order No., Start Date, Due Date, and Completed Date.
- **Apply Filters**: Use the sidebar to filter tasks by Order Number or Assignee, and toggle the "Show Delayed Tasks Only" option.
- **Switch Views**: Choose between the "Ideal Schedule" (projected timeline) and the "Actual Schedule" (based on completed dates).
- **Export Results**: Download the schedule as an Excel file using the provided download buttons.

## Deployment
The Task Management Dashboard can be deployed on **Streamlit Community Cloud** for easy access.

**Example Live URL**:  
`https://your-app-name.streamlit.app` (Replace with your actual app URL after deployment)

**Deployment Steps**:
1. Push your project to GitHub:
   ```bash
   git add .
   git commit -m "Deploy Task Management Dashboard"
   git push origin master
   ```

2. Sign in to [Streamlit Community Cloud](https://streamlit.io/cloud) using your GitHub account.

3. Click **New app**, select your repository (e.g., `yourusername/task-management-dashboard`), branch (`master`), and main file (`task_dashboard.py`).

4. Set a custom app URL (optional) and click **Deploy**.

Once deployed, you can share the generated URL with your team.

## Project Structure
```
task-management-dashboard/
│
├── task_dashboard.py  # Main application file
├── requirements.txt                   # List of dependencies
└── README.md                           # Project documentation
```

## Contributing
Contributions are welcome!  

To contribute:
1. Fork this repository.
2. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add your feature description"
   ```
4. Push to your branch:
   ```bash
   git push origin feature/your-feature-name
   ```
5. Open a pull request on GitHub.

## License
This project is open-source and available for use and modification.  
Please credit the original author and xAI when using this project.

## Contact
For questions or support, create an issue on the GitHub repository or reach out via xAI support channels.

## Acknowledgments
- Built using Streamlit, Pandas, Plotly, and NetworkX.
- Special thanks to Grok 3 (xAI) for development assistance.