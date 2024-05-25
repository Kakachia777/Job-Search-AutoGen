from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from crewai.tasks.task_output import TaskOutput
import json, requests, os

llm = ChatOpenAI(model="gpt-4-turbo-preview")

class JobSearchTools:
    @tool("Job Search Tool")
    def search_jobs(input_json: str) -> str:
        """Search for job listings using the Adzuna API."""
        # Parse input JSON string
        try:
            input_data = json.loads(input_json)
            role = input_data['role']
            location = input_data['location']
            num_results = input_data['num_results']
        except (json.JSONDecodeError, KeyError) as e:
            return """The tool accepts input in JSON format with the 
                    following schema: {'role': '<role>', 'location': '<location>', 'num_results': <number>}. 
                    Ensure to format the input accordingly."""

        app_id = os.getenv('ADZUNA_APP_ID')
        api_key = os.getenv('ADZUNA_API_KEY')
        base_url = "http://api.adzuna.com/v1/api/jobs"
        url = f"{base_url}/us/search/1?app_id={app_id}&app_key={api_key}&results_per_page={num_results}&what={role}&where={location}&content-type=application/json"
        
        try:
            response = requests.get(url)
            response.raise_for_status()  # This will raise an HTTPError if the HTTP request returned an unsuccessful status code.
            jobs_data = response.json()

            job_listings = []
            for job in jobs_data.get('results', []):
                job_details = f"Title: {job['title']}, Company: {job['company']['display_name']}, Location: {job['location']['display_name']}, Description: {job['description'][:100]}..."
                job_listings.append(job_details)
            return '\n'.join(job_listings)
        except requests.exceptions.HTTPError as err:
            raise ToolException(f"HTTP Error: {err}")
        except requests.exceptions.RequestException as e:
            raise ToolException(f"Error: {e}")

def callback_function(output: TaskOutput):
    with open("task_output.txt", "a") as file:
        file.write(f"""{output.result}\n\n""")
    print("Result saved to task_output.txt")

job_searcher_agent = Agent(
    role='Job Searcher',
    goal='Search for jobs in the field of interest, focusing on enhancing relevant skills',
    backstory="""You are actively searching for job opportunities in your field, ready to utilise and expand your skill set in a new role.""",
    verbose=True,
    llm=llm,
    allow_delegation=True,
    tools=[JobSearchTools().search_jobs]
)

skills_development_agent = Agent(
    role='Skills Development Advisor',
    goal='Identify key skills required for jobs of interest and advise on improving them',
    backstory="""As a skills development advisor, you assist job searchers in identifying crucial skills for their target roles and recommend ways to develop these skills.""",
    verbose=True,
    allow_delegation=True,
    llm=llm
)

interview_preparation_coach = Agent(
    role='Interview Preparation Coach',
    goal='Enhance interview skills, focusing on common questions, presentation, and communication',
    backstory="""Expert in coaching job searchers on successful interview techniques, including mock interviews and feedback.""",
    verbose=True,
    allow_delegation=True,
    llm=llm
)

career_advisor = Agent(
    role='Career Advisor',
    goal='Assist in resume building, LinkedIn profile optimization, and networking strategies',
    backstory="""Experienced in guiding candidates through their job search journey, offering personalized advice on career development and application processes.""",
    verbose=True,
    allow_delegation=True,
    llm=llm
)

# Define tasks for your agents
job_search_task = Task(
    description="""Search for current job openings for the Senior Data Scientist role in New York 
    using the Job Search tool. Find 5 vacant positions in total. 
    Emphasize the key skills required.
    The tool accepts input in JSON format with the 
    following schema: {'role': '<role>', 'location': '<location>', 'num_results': <number>}. 
    Ensure to format the input accordingly.""",
    agent=job_searcher_agent,
    tools=[JobSearchTools().search_jobs],
    callback=callback_function
)

skills_highlighting_task = Task(
    description="""Based on the identified job openings, list the key skills required for each position separately.
    Provide recommendations on how candidates can acquire or improve these skills through courses, self-study, or practical experience.""",
    agent=skills_development_agent,
    context=[job_search_task],
    callback=callback_function
)

interview_preparation_task = Task(
    description="""Prepare job searchers for interviews by conducting mock interviews and offering feedback on their responses, presentation, and communication skills, for each role separately.""",
    agent=interview_preparation_coach,
    context=[job_search_task],
    callback=callback_function
)

career_advisory_task = Task(
    description="""Offer guidance on resume building, optimizing LinkedIn profiles, and effective networking strategies to enhance job application success, for each role separately.""",
    agent=career_advisor,
    context=[job_search_task],
    callback=callback_function
)

# Set up your crew with a sequential process (tasks executed sequentially by default)
job_search_crew = Crew(
    agents=[job_searcher_agent, skills_development_agent, interview_preparation_coach, career_advisor],
    tasks=[job_search_task, skills_highlighting_task, interview_preparation_task, career_advisory_task],
    process=Process.hierarchical,
    manager_llm=llm,
)

# Initiate the crew to start working on its tasks
crew_result = job_search_crew.kickoff()

print(crew_result)