import pandas as pd
import plotly.express as px
import streamlit as st
import datetime
import logging
import networkx as nx
from io import BytesIO

# Set up logging
logging.basicConfig(filename='task_management.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Predefined dependencies and normalized activity names
dependencies = {
    "OCCR": [],
    "GAD Submission": ["OCCR"],
    "QAP Submission": ["OCCR"],
    "Procedure Submission": ["OCCR"],
    "GAD Approval": ["GAD Submission"],
    "QAP Approval": ["QAP Submission"],
    "Procedure Approval": ["Procedure Submission"],
    "BOM Release": ["GAD Approval"],
    "Casting Drawing Release": ["GAD Approval"],
    "MRP RUN": ["BOM Release"],
    "PO Release": ["MRP RUN"],
    "Machining Drawing Release": ["GAD Approval"],
    "Procurement Trim": ["PO Release"],
    "Procurement Casing Mtl": ["PO Release"],
    "Procurement Seals": ["PO Release"],
    "Procurement Fasteners": ["PO Release"],
    "Procurement Accessories": ["PO Release"],
    "Production Planning": ["Procurement Casing Mtl"],
    "Machining": ["Production Planning"],
    "Assembly": ["Machining"],
    "Testing": ["Assembly"],
    "Inspection": ["Testing"],
    "Dispatch": ["Inspection"]
}

activity_mapping = {
    "OCCR": "OCCR",
    "GAD": "GAD Submission",
    "GAD Submission": "GAD Submission",
    "QAP Submission": "QAP Submission",
    "Procedure Submission": "Procedure Submission",
    "GAD Approval": "GAD Approval",
    "QAP Approval": "QAP Approval",
    "Procedure Approval": "Procedure Approval",
    "BOM Release": "BOM Release",
    "Casting Drawing Release": "Casting Drawing Release",
    "MRP RUN": "MRP RUN",
    "PO Release": "PO Release",
    "M/C Drawing Release": "Machining Drawing Release",
    "Proc_Casing Mtl": "Procurement Casing Mtl",
    "Proc_Casing_Mtl": "Procurement Casing Mtl",
    "Proc_Seals": "Procurement Seals",
    "Proc_Fastners": "Procurement Fasteners",
    "Proc_Accessories": "Procurement Accessories",
    "Production_Planning": "Production Planning",
    "Machining": "Machining",
    "Assembly": "Assembly",
    "Testing": "Testing",
    "Inspection": "Inspection",
    "Dispatch": "Dispatch",
    "Seismic Analysis": "Seismic Analysis",
    "Seismic Approval": "Seismic Approval",
    "Procedure Approvals": "Procedure Approval",
    "GAD submission": "GAD Submission",
    "Datasheet Preperation": "Datasheet Preparation",
    "Forging Drawing Release": "Forging Drawing Release",
    "Proc_Trim_Mtl": "Procurement Trim"
}

def detect_cycles(dep_graph):
    try:
        G = nx.DiGraph(dep_graph)
        cycle = nx.find_cycle(G, orientation="original")
        return cycle
    except nx.NetworkXNoCycle:
        return None

def tag_fallback_source(row):
    for source in ['completed date', 'start date', 'due date']:
        if pd.notna(row.get(source)):
            return source
    return "fallback base"

def calculate_lead_time(activity, occr_start, dispatch_due, default_days=1):
    delta = (dispatch_due - occr_start).days if dispatch_due > occr_start else 0
    lead_rules = {
        "OCCR": 2,
        "GAD Submission": 0.05 * delta,
        "QAP Submission": 0.05 * delta,
        "Procedure Submission": 0.05 * delta,
        "GAD Approval": min(14, 0.10 * delta),
        "QAP Approval": min(14, 0.10 * delta),
        "Procedure Approval": 0.10 * delta,
        "BOM Release": 0.10 * delta,
        "Casting Drawing Release": min(7, 0.10 * delta),
        "MRP RUN": min(7, 0.10 * delta),
        "PO Release": 3,
        "Machining Drawing Release": max(7, 0.10 * delta),
        "Procurement Trim": 0.50 * delta,
        "Procurement Casing Mtl": 0.50 * delta,
        "Procurement Seals": 0.50 * delta,
        "Procurement Fasteners": 0.50 * delta,
        "Procurement Accessories": 0.50 * delta,
        "Production Planning": 2,
        "Machining": 0.15 * delta,
        "Assembly": 0.05 * delta,
        "Testing": 0.05 * delta,
        "Inspection": 0.02 * delta,
        "Dispatch": 1,
        "Seismic Analysis": 0.10 * delta,
        "Seismic Approval": 0.10 * delta,
        "Datasheet Preparation": 0.05 * delta,
        "Forging Drawing Release": 0.10 * delta
    }
    return pd.Timedelta(days=max(lead_rules.get(activity, default_days), 1))

def classify_task(row):
    now = pd.Timestamp.now()  # Timezone-naive to match input data
    if pd.isna(row['ideal completion']):
        return "No Data"
    if pd.notna(row.get('completed date')):
        if row['completed date'] <= row['ideal completion']:
            return "Completed On Time"
        else:
            return "Completed Late"
    else:
        if row['ideal completion'] >= now:
            return "Not Due Yet"
        else:
            return "Overdue"

def calculate_ideal_schedule(df):
    cycle = detect_cycles(dependencies)
    if cycle:
        st.warning(f"Dependency cycle detected: {cycle}. Schedule may be unreliable.")
        logging.warning(f"Dependency cycle detected: {cycle}")

    df = df.sort_values(['order no.', 'task no'])
    ideal_schedule = df.copy()
    ideal_schedule['ideal start'] = pd.NaT
    ideal_schedule['ideal completion'] = pd.NaT
    ideal_schedule['fallback source'] = None
    ideal_schedule['po date'] = pd.NaT
    ideal_schedule['dispatch due date'] = pd.NaT
    ideal_schedule['lead time (days)'] = 0
    ideal_schedule['Status'] = "No Data"

    for order in df['order no.'].unique():
        order_df = ideal_schedule[ideal_schedule['order no.'] == order].copy()
        occr_rows = order_df[order_df['activity name'] == 'OCCR']
        if occr_rows.empty:
            base_date = df[['start date', 'due date', 'completed date']].min().min()
            fallback = base_date if pd.notna(base_date) else pd.Timestamp('2025-01-01')
            logging.warning(f"No OCCR task found for order {order}, using fallback: {fallback}")
        else:
            base_date = pd.to_datetime(occr_rows['start date'].min()) if not occr_rows['start date'].isna().all() else pd.Timestamp('2025-01-01')
            fallback = base_date
        logging.info(f"Order {order}: Base date: {base_date}, Fallback: {fallback}")

        dispatch_rows = order_df[order_df['activity name'] == 'Dispatch']
        if dispatch_rows.empty:
            dispatch_due = base_date + pd.Timedelta(days=90)
        else:
            dispatch_due = pd.to_datetime(dispatch_rows['due date'].max()) if not dispatch_rows['due date'].isna().all() else base_date + pd.Timedelta(days=90)

        ideal_schedule.loc[ideal_schedule['order no.'] == order, 'po date'] = base_date
        ideal_schedule.loc[ideal_schedule['order no.'] == order, 'dispatch due date'] = dispatch_due

        G = nx.DiGraph()
        for idx, row in order_df.iterrows():
            G.add_node(idx)
            activity = row['activity name']
            deps = dependencies.get(activity, [])
            for dep in deps:
                dep_rows = order_df[order_df['activity name'] == dep]
                for dep_idx in dep_rows.index:
                    G.add_edge(dep_idx, idx)

        try:
            topo_order = list(nx.topological_sort(G))
        except nx.NetworkXUnfeasible:
            st.error(f"Dependency cycle detected within order {order}. Cannot schedule tasks.")
            logging.error(f"Dependency cycle within order {order}")
            continue

        for index in topo_order:
            row = order_df.loc[index]
            ideal_schedule.at[index, 'ideal start'] = pd.to_datetime(row['start date']) if pd.notna(row['start date']) else base_date
            ideal_schedule.at[index, 'fallback source'] = tag_fallback_source(row) if pd.isna(row['start date']) else 'start date'

        for index in topo_order:
            activity = order_df.loc[index, 'activity name']
            deps = dependencies.get(activity, [])
            if deps:
                dep_completions = []
                for dep in deps:
                    dep_rows = order_df[order_df['activity name'] == dep]
                    if not dep_rows.empty:
                        dep_completion = ideal_schedule.loc[dep_rows.index, 'ideal completion'].max()
                        if pd.notna(dep_completion):
                            dep_completions.append(dep_completion)
                if dep_completions:
                    ideal_start = max(dep_completions)
                    ideal_schedule.at[index, 'ideal start'] = ideal_start
                    ideal_schedule.at[index, 'fallback source'] = 'dependency'

            start = ideal_schedule.at[index, 'ideal start']
            if pd.notna(start):
                lead_time = calculate_lead_time(activity, base_date, dispatch_due)
                ideal_schedule.at[index, 'ideal completion'] = start + lead_time
                lead_time_days = (ideal_schedule.at[index, 'ideal completion'] - ideal_schedule.at[index, 'ideal start']).days
                ideal_schedule.at[index, 'lead time (days)'] = lead_time_days

        for index in topo_order:
            ideal_schedule.at[index, 'Status'] = classify_task(ideal_schedule.loc[index])

    return ideal_schedule

def calculate_slack(df):
    df = df.sort_values(['order no.', 'task no'])
    slack_df = df.copy()
    slack_df['latest start'] = pd.NaT
    slack_df['latest completion'] = pd.NaT
    slack_df['slack days'] = 0

    for order in df['order no.'].unique():
        order_df = slack_df[slack_df['order no.'] == order]
        dispatch_row = order_df[order_df['activity name'] == 'Dispatch']
        if dispatch_row.empty:
            latest_finish = df[['due date', 'completed date']].max().max()
            latest_finish = latest_finish if pd.notna(latest_finish) else pd.Timestamp.now() + pd.Timedelta(days=90)
        else:
            latest_finish = pd.to_datetime(dispatch_row['due date'].iloc[0]) if not pd.isna(dispatch_row['due date'].iloc[0]) else dispatch_row['ideal completion'].iloc[0]

        for index in order_df.index[::-1]:
            activity = order_df.loc[index, 'activity name']
            successors = [act for act, deps in dependencies.items() if activity in deps]
            filtered_successors = [succ for succ in successors if not order_df[order_df['activity name'] == succ].empty]
            successor_starts = []
            for succ in filtered_successors:
                succ_latest_start = order_df.loc[order_df['activity name'] == succ, 'latest start']
                if not succ_latest_start.empty and not succ_latest_start.isna().all():
                    successor_starts.append(succ_latest_start.iloc[0])
            
            if activity == "Dispatch":
                slack_df.at[index, 'latest completion'] = latest_finish
            else:
                latest_completion = min(successor_starts) if successor_starts else latest_finish
                slack_df.at[index, 'latest completion'] = latest_completion

            lead_time = (order_df.loc[index, 'ideal completion'] - order_df.loc[index, 'ideal start'])
            slack_df.at[index, 'latest start'] = slack_df.at[index, 'latest completion'] - lead_time

        for index in order_df.index:
            if pd.notna(slack_df.at[index, 'latest start']) and pd.notna(slack_df.at[index, 'ideal start']):
                slack = (slack_df.at[index, 'latest start'] - slack_df.at[index, 'ideal start']).days
                slack_df.at[index, 'slack days'] = max(slack, 0)

    return slack_df

def create_gantt_chart(df, type='ideal'):
    df = df.sort_values(['order no.', f'{type} start'])
    df['y_label'] = df['activity name']
    df['count'] = df.groupby(['order no.', 'activity name']).cumcount() + 1
    df['y_label'] = df.apply(lambda row: f"{row['activity name']}" if row['count'] == 1 else f"{row['activity name']} ({row['count']})", axis=1)
    df = df.drop(columns=['count'])
    
    if df.empty or df[f'{type} start'].isna().all() or df[f'{type} completion'].isna().all():
        st.warning("No valid data available for the Gantt chart.")
        return px.timeline()

    color_mapping = {
        "Not Due Yet": "lightblue",
        "Completed On Time": "green",
        "Overdue": "pink",
        "Completed Late": "yellow",
        "No Data": "grey"
    }

    fig = px.timeline(df, x_start=f'{type} start', x_end=f'{type} completion',
                      y='y_label', color='Status',
                      color_discrete_map=color_mapping,
                      title=f'{type.capitalize()} Schedule',
                      hover_data=['task no', 'activity name', 'assigned to', 'fallback source', 'slack days', 'completed date'])
    
    fig.update_yaxes(categoryorder='array', categoryarray=df['y_label'].tolist()[::-1])
    fig.update_layout(
        xaxis_range=[df[f'{type} start'].min() - pd.Timedelta(days=7),
                     df[f'{type} completion'].max() + pd.Timedelta(days=7)],
        xaxis_title="Date", yaxis_title="Task Details",
        width=1200, height=600,
        yaxis=dict(domain=[0, 1], tickfont=dict(size=10)),
        xaxis=dict(domain=[0.15, 0.95]),
        margin=dict(l=150, r=20, t=50, b=50)
    )
    fig.update_traces(width=0.6)
    return fig

def export_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Schedule Data', index=False)
    output.seek(0)
    return output

def calculate_actual_schedule(df):
    actual_schedule = df.copy()
    actual_schedule['actual start'] = pd.to_datetime(df['start date'], errors='coerce')
    actual_schedule['actual completion'] = pd.to_datetime(df['completed date'], errors='coerce')
    actual_schedule['fallback source'] = actual_schedule.apply(tag_fallback_source, axis=1)
    return actual_schedule

def load_data(file):
    try:
        df = pd.read_excel(file)
        df.columns = [col.strip().lower() for col in df.columns]
        
        column_mappings = {
            'task name': 'activity name',
            'task id': 'task no',
            'bucket name': 'order no.'
        }
        for old_col, new_col in column_mappings.items():
            if old_col in df.columns and new_col not in df.columns:
                df = df.rename(columns={old_col: new_col})
                logging.info(f"Renamed column '{old_col}' to '{new_col}'")

        date_columns = ['created date', 'start date', 'due date', 'completed date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')
        
        if 'activity name' not in df.columns:
            st.error(f"Column 'activity name' not found. Available columns: {df.columns.tolist()}")
            return None
        df['activity name'] = df['activity name'].map(activity_mapping).fillna(df['activity name'])
        
        required_columns = ['task no', 'activity name', 'order no.', 'start date', 'due date', 'completed date']
        if 'assigned to' not in df.columns:
            df['assigned to'] = 'Unassigned'
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}. Available columns: {df.columns.tolist()}")
            return None
        return df
    except Exception as e:
        logging.error(f"Error loading file: {e}")
        st.error(f"Error loading file: {e}. Please check the log file for details.")
        return None

def main():
    st.title("📊 Task Management Dashboard")
    uploaded_file = st.file_uploader("Upload your Microsoft Planner Excel file", type="xlsx")

    if uploaded_file:
        df = load_data(uploaded_file)
        if df is not None:
            ideal_df = calculate_ideal_schedule(df)
            ideal_df_with_slack = calculate_slack(ideal_df)
            actual_df = calculate_actual_schedule(df)

            st.sidebar.header("Filters")
            order_filter = st.sidebar.multiselect("Filter by Order No.", options=df['order no.'].unique(), default=df['order no.'].unique())
            assignee_filter = st.sidebar.multiselect("Filter by Assignee", options=df['assigned to'].dropna().unique(), default=df['assigned to'].dropna().unique())
            delayed_filter = st.sidebar.checkbox("Show Delayed Tasks Only", value=False)

            filtered_ideal = ideal_df_with_slack[ideal_df_with_slack['order no.'].isin(order_filter) & ideal_df_with_slack['assigned to'].isin(assignee_filter)]
            filtered_actual = actual_df[actual_df['order no.'].isin(order_filter) & actual_df['assigned to'].isin(assignee_filter)]

            if delayed_filter:
                now = pd.Timestamp.now()
                filtered_ideal = filtered_ideal[filtered_ideal['ideal completion'] < filtered_ideal['completed date'].fillna(now)]
                filtered_actual = filtered_actual[filtered_actual['actual completion'] < filtered_actual['completed date'].fillna(now)]

            view = st.radio("Select View", ("Ideal Schedule", "Actual Schedule"))

            if view == "Ideal Schedule":
                st.plotly_chart(create_gantt_chart(filtered_ideal, "ideal"), use_container_width=True)
                st.subheader("📅 Ideal Schedule Details")
                display_df = filtered_ideal.sort_values(['order no.', 'ideal start'])[[ 
                    'order no.', 'task no', 'activity name', 'ideal start', 'ideal completion',
                    'po date', 'dispatch due date', 'lead time (days)', 'slack days', 'fallback source', 'Status'
                ]]
                st.dataframe(display_df)
                excel_file = export_to_excel(display_df)
                st.download_button("📥 Download Ideal Schedule", excel_file, "ideal_schedule.xlsx")
            else:
                st.plotly_chart(create_gantt_chart(filtered_actual, "actual"), use_container_width=True)
                st.subheader("📅 Actual Schedule Details")
                st.dataframe(filtered_actual[[
                    'order no.', 'task no', 'activity name', 'actual start', 'actual completion', 'fallback source'
                ]])
                excel_file = export_to_excel(filtered_actual)
                st.download_button("📥 Download Actual Schedule", excel_file, "actual_schedule.xlsx")

if __name__ == '__main__':
    main()