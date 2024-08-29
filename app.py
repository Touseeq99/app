import streamlit as st
import pandas as pd
import re
from crewai import Agent, Task, Crew
from crewai_tools import  SerperDevTool, WebsiteSearchTool, CSVSearchTool
import os
from io import StringIO

# Set up API keys securely
os.environ["SERPER_API_KEY"] = "1cf2a1aabd5375b4c8181063dbafdd737a301ad6"
os.environ["OPENAI_API_KEY"] = "sk-5GSD4deAY3It4kTp8b1mVpQAaTh_ILOr_7boYda5K7T3BlbkFJsi2IXzPwtT9sutOWyENQ7qiCzrT-z-an-mN21hswQA"

def parse_task_output(raw_output):
    # Regular expressions for extracting job title, location, and company
    title_pattern = r'Job Title:\s*(.*)'
    location_pattern = r'Job Location:\s*(.*)'
    company_pattern = r'Company:\s*(.*)'

    # Search for matches
    title_match = re.search(title_pattern, raw_output)
    location_match = re.search(location_pattern, raw_output)
    company_match = re.search(company_pattern, raw_output)

    # Extract matched groups or set as 'Not specified'
    title = title_match.group(1) if title_match else 'Not specified'
    location = location_match.group(1) if location_match else 'Not specified'
    company = company_match.group(1) if company_match else 'Not specified'

    return title, location, company

def process_csv(file):
    # Read the dataset
    df = pd.read_csv(file, encoding="latin1")
    websites = df["Company Name"].tolist()

    # Instantiate tools
    file_tool = CSVSearchTool()
    web_rag_tool = WebsiteSearchTool()
    serp = SerperDevTool()

    # Initialize an empty list to store tasks
    tasks = []

    # Loop over each company name in websites
    for item in websites:
        # Define researcher agent
        researcher = Agent(
            role='Job Expert',
            goal=f'Find job titles, locations, and job responsibilities for mid to senior level positions at {item} posted in the last three months on job boards, company career pages, and social media websites.',
            backstory=f'You are a Business Development Manager with 25 years of experience helping to identify hiring needs for {item}.',
            tools=[serp],
            verbose=True
        )

        # Define the research task
        research = Task(
            description=f'Find job titles and locations for mid to senior level positions posted by {item} in the last three months on job boards, company career pages, and social media websites.',
            expected_output='Output just one job title and job location with the company name. Only one job title, job location, and company is required',
            agent=researcher
        )

        # Add tasks to list
        tasks.extend([research])

    # Assemble the crew
    crew = Crew(
        agents=[researcher],
        tasks=tasks,
        verbose=True,
        planning=True  # Enable planning feature
    )

    # Execute tasks
    output = crew.kickoff()

    # Initialize lists to store structured data
    job_titles = []
    job_locations = []
    companies = []

    # Process each task output
    for task_output in output.tasks_output:
        raw_output = task_output.raw
        title, location, company = parse_task_output(raw_output)
        job_titles.append(title)
        job_locations.append(location)
        companies.append(company)

    # Create a DataFrame
    df_jobs = pd.DataFrame({
        'Job Title': job_titles,
        'Job Location': job_locations,
        'Company': companies
    })

    return df_jobs

# Streamlit UI
st.title('Job Listings Extractor')

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    with st.spinner('Processing...'):
        df_jobs = process_csv(uploaded_file)
        st.success('Processing Done!')
        
        st.write("Job Listings:")
        st.dataframe(df_jobs)

        # Convert DataFrame to CSV for download
        csv = df_jobs.to_csv(index=False)
        st.download_button(
            label="Download job_listings.csv",
            data=csv,
            file_name='job_listings.csv',
            mime='text/csv'
        )
